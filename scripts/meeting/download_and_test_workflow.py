#!/usr/bin/env python3
"""下载完整 Workflow 模型，并对仓库根目录的三段 M4A 做端到端测试。

默认执行：
  1. Qwen3-ASR-0.6B-4bit (Hugging Face cache)
  2. Silero VAD (models/silero_vad.onnx)
  3. FunASR ct-punc-c (ModelScope cache)
  4. resemblyzer 声纹模型（首次构造 VoiceEncoder 时下载）
  5. 对根目录排序后的三个 *.m4a 执行 VAD -> ASR -> 标点 -> SD

使用 --skip-download 可在模型准备好后只重复测试。完整测试可能需要较长时间，
结果写入 /tmp/astra-workflow-test/，不会污染仓库。
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

QWEN_REPO = "mlx-community/Qwen3-ASR-0.6B-4bit"
# Current FunASR/ModelScope compact ct-punc model (~292MB).
PUNC_REPO = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
VAD_PATH = PROJECT_ROOT / "models" / "silero_vad.onnx"
RECORDINGS = (
    PROJECT_ROOT / "1787882095007-5ef1.m4a",
    PROJECT_ROOT / "1787899449766-9e93.m4a",
    PROJECT_ROOT / "1788504363364-1819.m4a",
)


@dataclass
class RecordingResult:
    filename: str
    duration_s: float
    elapsed_s: float
    segments: int
    speakers: list[str]
    text: str
    error: str = ""


def _download_url(url: str, destination: Path) -> Path:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 100_000:
        print(f"[model] VAD already present: {destination} ({destination.stat().st_size} bytes)")
        return destination
    part = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            print(f"[model] downloading VAD ({attempt}/3): {url}")
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with part.open("wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
            part.replace(destination)
            print(f"[model] VAD ready: {destination} ({destination.stat().st_size} bytes)")
            return destination
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")


def _download_models() -> str:
    from huggingface_hub import snapshot_download

    print(f"[model] downloading/checking {QWEN_REPO}")
    qwen_path = snapshot_download(repo_id=QWEN_REPO, resume_download=True)
    print(f"[model] Qwen3 ready: {qwen_path}")
    _download_url(VAD_URL, VAD_PATH)

    try:
        from modelscope import snapshot_download as ms_snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is required to download the FunASR punctuation model; "
            "install backend/requirements.txt first"
        ) from exc
    print(f"[model] downloading/checking {PUNC_REPO}")
    punc_path = ms_snapshot_download(PUNC_REPO)
    print(f"[model] punctuation ready: {punc_path}")
    return str(punc_path)


def _ensure_dependencies() -> None:
    missing = [
        name for name in ("soundfile", "resemblyzer", "funasr", "modelscope")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "missing runtime packages: " + ", ".join(missing)
            + ". Run: .venv/bin/pip install -r backend/requirements.txt"
        )


def _duration(path: Path) -> float:
    output = subprocess.run(
        ["/usr/bin/afinfo", str(path)], capture_output=True, text=True, check=True
    ).stdout
    for line in output.splitlines():
        if "estimated duration:" in line.lower():
            return float(line.split(":", 1)[1].strip().split()[0])
    raise RuntimeError(f"cannot determine duration: {path}")


async def _run_recording(pipe: Any, path: Path, output_dir: Path) -> RecordingResult:
    started = time.perf_counter()
    try:
        wav, duration = await asyncio.to_thread(pipe.decode_to_wav, path)
        try:
            language, segments = await pipe.transcribe(wav)
        finally:
            Path(wav).unlink(missing_ok=True)
            Path(wav).parent.rmdir()
        speakers = sorted({segment.speaker for segment in segments if segment.speaker})
        text = "\n".join(
            f"[{segment.start:08.2f}-{segment.end:08.2f}] "
            f"{segment.speaker or '?'} {segment.text}"
            for segment in segments
        )
        (output_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        return RecordingResult(
            filename=path.name,
            duration_s=duration,
            elapsed_s=time.perf_counter() - started,
            segments=len(segments),
            speakers=speakers,
            text=text,
        )
    except Exception as exc:
        return RecordingResult(
            filename=path.name,
            duration_s=_duration(path),
            elapsed_s=time.perf_counter() - started,
            segments=0,
            speakers=[],
            text="",
            error=f"{exc.__class__.__name__}: {exc}",
        )


async def _test_workflow(punctuation_model: str, output_dir: Path) -> list[RecordingResult]:
    from app.config import Settings
    from app.core.meeting import MeetingPipeline
    from app.core.workflow import ResemblyzerDiarizationStage
    from app.models.asr_client import MlxAudioAsrClient
    from app.models.llm_client import OpenAICompatLLMClient
    from app.models.punctuation_client import FunASRPunctuationClient

    settings = Settings.from_env()
    asr = MlxAudioAsrClient(
        settings.asr_model,
        settings.asr_language,
        max_tokens=settings.asr_max_tokens,
        repetition_penalty=settings.asr_repetition_penalty,
        repetition_context_size=settings.asr_repetition_context_size,
        chunk_duration=settings.asr_chunk_duration_seconds,
        long_audio_threshold=settings.asr_long_audio_threshold_seconds,
        hotwords=(),
        system_prompt="",
    )
    punctuation = FunASRPunctuationClient(
        punctuation_model, device=settings.punctuation_device
    )
    llm = OpenAICompatLLMClient(
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key,
        request_timeout_seconds=600,
        connect_timeout_seconds=5,
        stream_idle_timeout_seconds=120,
    )
    pipe = MeetingPipeline(
        llm=llm,
        asr=asr,
        vad_model=str(VAD_PATH),
        punctuation=punctuation,
        diarization=ResemblyzerDiarizationStage(),
    )
    results: list[RecordingResult] = []
    try:
        for recording in RECORDINGS:
            print(f"\n[test] {recording.name} ({_duration(recording):.1f}s)")
            result = await _run_recording(pipe, recording, output_dir)
            results.append(result)
            if result.error:
                print(f"[test] FAIL {result.error}")
            else:
                print(
                    f"[test] OK segments={result.segments} speakers={result.speakers} "
                    f"elapsed={result.elapsed_s:.1f}s"
                )
    finally:
        await llm.aclose()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/astra-workflow-test")
    )
    args = parser.parse_args()
    missing_recordings = [path for path in RECORDINGS if not path.is_file()]
    if missing_recordings:
        print(f"missing recordings: {missing_recordings}", file=sys.stderr)
        return 2
    try:
        _ensure_dependencies()
        punc_model = _download_models() if not args.skip_download else PUNC_REPO
        if not VAD_PATH.is_file():
            raise RuntimeError(f"missing {VAD_PATH}; rerun without --skip-download")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results = asyncio.run(_test_workflow(punc_model, args.output_dir))
        report = args.output_dir / "report.json"
        report.write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 1 if any(result.error for result in results) else 0
    except Exception as exc:
        print(f"FATAL: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
