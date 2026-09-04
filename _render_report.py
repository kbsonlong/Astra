#!/usr/bin/env python3
"""渲染完整会议报告: 转写全文(带speaker) + AI纪要 + 翻译 -> md"""
import asyncio, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from app.config import Settings
from app.core.meeting import MeetingPipeline, Segment
from app.models.asr_client import MlxAudioAsrClient
from app.models.llm_client import OpenAICompatLLMClient

SRC = Path("/Users/kbsonlong/Astra/1788504363364-1819.m4a")
CACHE = Path("/tmp/meeting_1819_segments.json")


def fmt(sec: float) -> str:
    return f"{int(sec//60):02d}:{int(sec%60):02d}"


async def main() -> None:
    s = Settings.from_env()
    llm = OpenAICompatLLMClient(s.llm_base_url, s.llm_model, s.llm_api_key,
                                 request_timeout_seconds=300, connect_timeout_seconds=5,
                                 stream_idle_timeout_seconds=90)
    asr = MlxAudioAsrClient(s.asr_model, s.asr_language,
                            max_tokens=s.asr_max_tokens,
                            repetition_penalty=s.asr_repetition_penalty,
                            repetition_context_size=s.asr_repetition_context_size,
                            hotwords=(), system_prompt="")
    pipe = MeetingPipeline(llm=llm, asr=asr, vad_model="models/silero_vad.onnx")

    d = json.loads(CACHE.read_text(encoding="utf-8"))
    segs = [Segment(x["start"], x["end"], x["text"]) for x in d["segments"]]
    wav, dur = await asyncio.to_thread(pipe.decode_to_wav, SRC)
    await pipe.diarize(wav, segs)

    # 转写全文
    tlines = []
    for x in segs:
        sp = x.speaker or "?"
        tlines.append(f"[{fmt(x.start)}] {sp} {x.text.strip()}")
    transcript = "\n".join(tlines)

    # 纪要
    t0 = time.time()
    summary, translation = await pipe.summarize(segs, meeting_topic="云资源成本优化")
    print(f"summarize {time.time()-t0:.0f}s")

    md = []
    md.append("# 会议纪要 · 云资源成本优化")
    md.append("")
    md.append(f"- 录音: {SRC.name} · 时长 {int(dur//60)}分{int(dur%60)}秒 · 简体中文")
    md.append(f"- 说话人: {sorted({x.speaker for x in segs if x.speaker})} · 引擎: Silero VAD + Qwen3-ASR + resemblyzer")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🤖 AI 纪要")
    md.append("")
    md.append(summary)
    md.append("")
    md.append("## 🌐 纪要翻译")
    md.append("")
    md.append(translation or "(未生成)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 🎙️ 转写全文（按说话人标注）")
    md.append("")
    md.append("```text")
    md.append(transcript)
    md.append("```")
    md.append("")

    out = Path("/tmp/meeting_report_full.md")
    out.write_text("\n".join(md), encoding="utf-8-sig")
    print(f"written {out} ({sum(len(m) for m in md)} chars)")
    # 打印纪要+翻译部分(不含全文)供直接展示
    sep = "\n---\n## 🎙️"
    body = "\n".join(md).split(sep)[0]
    print("===BODY===")
    print(body)


if __name__ == "__main__":
    asyncio.run(main())
