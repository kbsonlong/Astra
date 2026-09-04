#!/usr/bin/env python3
"""快速 diarize 冒烟: 44s 短样本 (2人对话), 不跑 LLM。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from app.core.meeting import MeetingPipeline

WHISPER_TURBO = "/Users/kbsonlong/.cache/modelscope/hub/models/mlx-community/whisper-large-v3-turbo"


async def main() -> None:
    pipe = MeetingPipeline(whisper_model=WHISPER_TURBO, llm=None)
    src = Path("/Users/kbsonlong/Astra/1787899449766-9e93.m4a")
    result = await pipe.process(src, filename=src.name, summarize=False)
    print(f"duration={result.duration_s:.1f}s segments={len(result.segments)}")
    print("speakers:", sorted({s.speaker for s in result.segments}))
    for s in result.segments[:12]:
        print(f"  [{s.start:5.1f}-{s.end:5.1f}] {s.speaker:3s} {s.text.strip()}")


if __name__ == "__main__":
    asyncio.run(main())
