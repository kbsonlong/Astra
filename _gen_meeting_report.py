#!/usr/bin/env python3
"""把 MeetingResult 渲染成可交付 markdown 报告 (API 同款逻辑, 供脚本直用)。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))


def render(result, topic: str = "", llm_model: str = "") -> str:
    out: list[str] = []
    out.append("# 会议纪要")
    out.append("")
    dur = result.duration_s
    speakers = sorted({s.speaker for s in result.segments if s.speaker})
    out.append(
        f"- **录音**: {result.filename} · 时长 {int(dur//60)}分{int(dur%60)}秒 · "
        f"语言 {result.language} · 说话人 {len(speakers)} 位: {'/'.join(speakers) if speakers else '未分离'}"
    )
    if topic:
        out.append(f"- **主题**: {topic}")
    out.append(f"- **生成**: {time.strftime('%Y-%m-%d %H:%M')}")
    out.append("")
    if result.summary:
        out += ["---", "", "## 🤖 AI 摘要", "", result.summary, ""]
    if result.translation:
        out += ["---", "", "## 🌐 全文翻译", "", result.translation, ""]
    out += ["---", "", "## 🎙️ 逐字稿（按说话人标注）", "", "```text", result.timeline_text(), "```", ""]
    return "\n".join(out)


def main() -> None:
    import asyncio

    from app.config import Settings
    from app.core.meeting import MeetingPipeline
    from app.models.llm_client import OpenAICompatLLMClient

    src = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/kbsonlong/Astra/1788504363364-1819.m4a")
    s = Settings.from_env()
    llm = OpenAICompatLLMClient(
        s.llm_base_url, s.llm_model, s.llm_api_key,
        request_timeout_seconds=300, connect_timeout_seconds=5, stream_idle_timeout_seconds=90,
    )
    pipe = MeetingPipeline(whisper_model=s.whisper_model_path, llm=llm)
    result = asyncio.run(pipe.process(src, filename=src.name, topic="云资源成本优化", summarize=True))
    md = render(result, topic="云资源成本优化", llm_model=s.llm_model)
    out = Path("/Users/kbsonlong/Astra/meetings")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{src.stem}.report.md"
    p.write_text(md, encoding="utf-8")
    print(f"written: {p} ({len(md)} chars)")
    # JSON 摘要供调试
    (out / f"{src.stem}.json").write_text(
        json.dumps({
            "filename": result.filename, "duration_s": result.duration_s,
            "language": result.language, "segments": len(result.segments),
            "speakers": sorted({x.speaker for x in result.segments if x.speaker}),
            "summary": result.summary[:800], "translation": result.translation[:800],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "segments": len(result.segments),
        "speakers": sorted({x.speaker for x in result.segments if x.speaker}),
        "summary_len": len(result.summary), "translation_len": len(result.translation),
    }))


if __name__ == "__main__":
    main()
