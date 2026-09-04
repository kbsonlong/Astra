#!/usr/bin/env python3
"""验证 diarize 时间轴合理性: 段标签连续性与时长分布。"""
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from app.core.meeting import MeetingPipeline, Segment

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/kbsonlong/Astra/1788504363364-1819.m4a")


async def main() -> None:
    pipe = MeetingPipeline()
    d = json.load(open("/tmp/meeting_1819_segments.json", encoding="utf-8"))
    segs = [Segment(s["start"], s["end"], s["text"]) for s in d["segments"]]
    wav, _ = await asyncio.to_thread(pipe.decode_to_wav, SRC)
    await pipe.diarize(wav, segs)

    runs = 0
    for i in range(1, len(segs)):
        if segs[i].speaker != segs[i - 1].speaker:
            runs += 1
    dur = defaultdict(float)
    for s in segs:
        dur[s.speaker] += s.duration
    print(f"总段数={len(segs)} 说话人切换次数={runs} "
          f"平均每段时长={sum(s.duration for s in segs)/len(segs):.1f}s")
    for sp in sorted(dur):
        print(f"  {sp}: {dur[sp]/60:.1f}min")
    # 抽样看前 40 段时间轴
    print("\n前 40 段时间轴:")
    for s in segs[:40]:
        print(f"  [{s.start:6.1f}] {s.speaker} {s.text.strip()[:40]}")
    # 切到 S2 的连续块(验证 S2 是否成段出现而非孤立)
    print("\nS2 连续块(≥3段):")
    blocks, cur = [], []
    for s in segs:
        if s.speaker == "S2":
            cur.append(s)
        elif cur:
            if len(cur) >= 3:
                blocks.append((cur[0].start, cur[-1].end, len(cur)))
            cur = []
    if cur and len(cur) >= 3:
        blocks.append((cur[0].start, cur[-1].end, len(cur)))
    for b in blocks[:8]:
        print(f"  [{b[0]/60:.1f}-{b[1]/60:.1f}min] {b[2]}段")


if __name__ == "__main__":
    asyncio.run(main())
