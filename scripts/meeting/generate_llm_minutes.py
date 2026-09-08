#!/usr/bin/env python3
"""基于 Workflow 已生成的逐字稿调用 LLM，输出中文会议纪要。

默认读取 /tmp/astra-workflow-test/report.json，不重新执行 VAD、ASR 或 SD。
默认使用本机缓存的 MLX Qwen2.5-7B-Instruct-4bit；本脚本只生成摘要，不生成翻译。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

LINE_RE = re.compile(
    r"^\[(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)\]\s+"
    r"(?P<speaker>\S+)\s?(?P<text>.*)$"
)
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
# 已确认的会议专有人名纠错；原始 ASR 逐字稿保持不变。
TRANSCRIPT_CORRECTIONS = {
    "宗师": "忠思",
    "下限了五台机器": "下线了五台机器",
    "冷资源中心": "云资源中心",
    "光单": "关单",
    "空单": "工单",
    "用更长的资源支撑更多的业务": "用更少的资源支撑更多的业务",
    "一个谷歌表哥": "一个 Google 表格",
    "你可以嫁给豆包": "你可以交给豆包",
}


class MlxLocalLLMClient:
    """将 mlx-lm 本地模型适配为 MeetingPipeline 所需的流式接口。"""

    def __init__(self, model_path: Path) -> None:
        self.model = str(model_path)
        self._model = None
        self._tokenizer = None

    def _load(self):
        from mlx_lm import load

        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = load(self.model)
        return self._model, self._tokenizer

    async def stream_chat(self, messages, *, temperature=0.2, top_p=1.0,
                          max_tokens=None, chat_template_kwargs=None):
        del chat_template_kwargs
        model, tokenizer = await asyncio.to_thread(self._load)
        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        text = await asyncio.to_thread(
            self._generate, model, tokenizer, prompt, temperature, top_p,
            max_tokens or 1024,
        )
        if text:
            yield text.strip()

    @staticmethod
    def _generate(model, tokenizer, prompt, temperature, top_p, max_tokens):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p)
        return generate(
            model,
            tokenizer,
            prompt,
            verbose=False,
            max_tokens=max_tokens,
            sampler=sampler,
        )

    async def aclose(self) -> None:
        return None


def _resolve_local_model(repo_id: str) -> Path:
    explicit_path = Path(repo_id).expanduser()
    if explicit_path.is_dir():
        return explicit_path
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"本地模型未缓存: {repo_id}; 请先下载到 Hugging Face cache"
        ) from exc


def _load_segments(report: Path):
    from app.core.workflow import Segment

    items = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("report must contain a list of recording results")
    loaded: list[tuple[str, float, list[Segment]]] = []
    for item in items:
        segments: list[Segment] = []
        for line in item.get("text", "").splitlines():
            match = LINE_RE.match(line)
            if match is None:
                continue
            segments.append(
                Segment(
                    float(match.group("start")),
                    float(match.group("end")),
                    _correct_transcript(match.group("text")),
                    match.group("speaker"),
                )
            )
        if not segments:
            raise ValueError(f"no transcript segments found for {item.get('filename')}")
        loaded.append((item["filename"], float(item["duration_s"]), segments))
    return loaded


def _correct_transcript(text: str) -> str:
    for source, target in TRANSCRIPT_CORRECTIONS.items():
        text = text.replace(source, target)
    return text


async def _generate(report: Path, output_dir: Path, topic: str, model_repo: str) -> None:
    from app.core.meeting import MeetingPipeline

    model_path = _resolve_local_model(model_repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    llm = MlxLocalLLMClient(model_path)
    pipeline = MeetingPipeline(llm=llm, asr=None, diarization=None)
    try:
        for filename, duration, segments in _load_segments(report):
            print(f"[llm] generating {filename} ({len(segments)} segments)")
            corrected_transcript = "\n".join(
                f"[{segment.start:08.2f}-{segment.end:08.2f}] "
                f"{segment.speaker or '?'} {segment.text}"
                for segment in segments
            )
            (output_dir / f"{Path(filename).stem}.corrected-transcript.txt").write_text(
                corrected_transcript + "\n", encoding="utf-8"
            )
            summary, _ = await pipeline.summarize(
                segments,
                meeting_topic=topic,
                translate=False,
            )
            output = output_dir / f"{Path(filename).stem}.minutes.md"
            output.write_text(
                "\n".join(
                    [
                        "# 会议纪要",
                        "",
                        f"- **录音**: {filename}",
                        f"- **时长**: {int(duration // 60)}分{int(duration % 60)}秒",
                        f"- **纪要模型**: {model_repo}",
                        *( [f"- **主题**: {topic}"] if topic else [] ),
                        "",
                        "---",
                        "",
                        summary,
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            print(f"[llm] written {output}")
    finally:
        await llm.aclose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path, default=Path("/tmp/astra-workflow-test/report.json")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/astra-workflow-test/minutes")
    )
    parser.add_argument("--topic", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="本地 MLX 模型目录或 Hugging Face repo")
    args = parser.parse_args()
    if not args.report.is_file():
        print(f"missing report: {args.report}", file=sys.stderr)
        return 2
    try:
        asyncio.run(_generate(args.report, args.out, args.topic, args.model))
    except Exception as exc:
        print(f"FATAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
