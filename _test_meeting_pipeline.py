#!/usr/bin/env python3
"""MeetingPipeline 集成测试 v2: 38.7min 会议, 转写结果缓存到 JSON。

用法:
  python _test_meeting_pipeline.py            # 全流程(转写缓存后可重跑)
  python _test_meeting_pipeline.py --cached   # 只跑 分离+纪要(用缓存转写)
"""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.config import Settings
from app.core.meeting import MeetingPipeline, Segment
from app.models.asr_client import MlxAudioAsrClient
from app.models.llm_client import OpenAICompatLLMClient

SRC = Path("/Users/kbsonlong/Astra/1788504363364-1819.m4a")
CACHE = Path("/tmp/meeting_1819_segments.json")


def dump_segments(segs: list[Segment], lang: str) -> None:
    CACHE.write_text(json.dumps({
        "language": lang,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segs],
    }, ensure_ascii=False), encoding="utf-8")


def load_segments() -> tuple[str, list[Segment]]:
    d = json.loads(CACHE.read_text(encoding="utf-8"))
    return d["language"], [Segment(s["start"], s["end"], s["text"]) for s in d["segments"]]


async def main() -> None:
    s = Settings.from_env()
    llm = OpenAICompatLLMClient(
        s.llm_base_url, s.llm_model, s.llm_api_key,
        request_timeout_seconds=300, connect_timeout_seconds=5,
        stream_idle_timeout_seconds=90,
    )
    asr = MlxAudioAsrClient(
        s.asr_model, s.asr_language,
        max_tokens=s.asr_max_tokens,
        repetition_penalty=s.asr_repetition_penalty,
        repetition_context_size=s.asr_repetition_context_size,
        hotwords=(),          # 会议转写忠实输出, 不带热词
        system_prompt="",
    )
    pipe = MeetingPipeline(llm=llm, asr=asr, vad_model="models/silero_vad.onnx")

    if "--cached" in sys.argv and CACHE.exists():
        lang, segs = load_segments()
        print(f"[cached] {len(segs)} segments, {lang}")
    else:
        wav, dur = await asyncio.to_thread(pipe.decode_to_wav, SRC)
        t0 = time.time()
        lang, segs = await pipe.transcribe(wav)
        print(f"[transcribe] {dur/60:.1f}min audio in {time.time()-t0:.1f}s -> {len(segs)} segs")
        dump_segments(segs, lang)

    t0 = time.time()
    # diarize 需要 wav; 每次重解码一次(afconvert ~5s)
    wav, dur = await asyncio.to_thread(pipe.decode_to_wav, SRC)
    await pipe.diarize(wav, segs)
    print(f"[diarize] {time.time()-t0:.1f}s")

    speakers = sorted({x.speaker for x in segs if x.speaker})
    from collections import Counter, defaultdict
    dur_by = defaultdict(float)
    for x in segs:
        dur_by[x.speaker] += x.duration
    print(f"speakers: {speakers}")
    for sp in speakers:
        print(f"  {sp}: {dur_by[sp]/60:.1f}min")

    t0 = time.time()
    result_summary, result_translation = await pipe.summarize(
        segs, meeting_topic="云资源成本优化", target_language="简体中文"
    )
    print(f"[llm summary+translate] {time.time()-t0:.1f}s")
    print("\n=== 摘要 ===\n")
    print(result_summary[:3000])
    if result_translation:
        print("\n=== 翻译(前800字) ===\n")
        print(result_translation[:800])

    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
