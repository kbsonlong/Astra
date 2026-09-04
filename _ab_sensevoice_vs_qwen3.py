#!/usr/bin/env python3
"""A/B: SenseVoice(VAD分段) vs 缓存 Qwen3 转写 —— 38.7min 会议样本。

对比方式: 同一 Silero VAD 分段(meeting.py 方案A 架构), 仅替换逐段转写引擎。
指标: 段覆盖 / 归一化精确匹配 / 去标点内容相似度 / 字符量差异 / RTF。
"""
import asyncio
import difflib
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from app.core.meeting import MeetingPipeline  # noqa: E402
from app.models.asr_client import SherpaSenseVoiceAsrClient  # noqa: E402

SRC = Path("/Users/kbsonlong/Astra/1788504363364-1819.m4a")
QWEN_CACHE = Path("/tmp/meeting_1819_segments.json")
# 本轮: language 固定 zh (auto_language=False), 排除 auto-language 漂移
LANG = "zh"
SV_CACHE = Path(f"/tmp/meeting_1819_sv_{LANG}_segments.json")
SV_AUTO_CACHE = Path("/tmp/meeting_1819_sv_segments.json")

_PUNCT = set("，。！？、；：""''（）…—·《》【】\t ")


def norm(t: str) -> str:
    return re.sub(r"\s+", "", t)


def norm_nopunct(t: str) -> str:
    return "".join(ch for ch in norm(t) if ch not in _PUNCT)


async def main() -> None:
    sv = SherpaSenseVoiceAsrClient(
        model_dir="models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
        num_threads=2,
        provider="cpu",
        auto_language=False,  # 固定语言, 排除 auto 漂移
        language=LANG,
        use_itn=True,
    )
    pipe = MeetingPipeline(asr=sv, llm=None, mt_llm=None)

    print("[1/3] 解码音频...", flush=True)
    t0 = time.time()
    wav, dur = await asyncio.to_thread(pipe.decode_to_wav, SRC)
    print(f"      解码完成 {dur:.1f}s 音频, 耗时 {time.time()-t0:.1f}s", flush=True)

    print("[2/3] Silero VAD 分段 + SenseVoice 逐段转写...", flush=True)
    t1 = time.time()
    lang, segs = await pipe.transcribe(wav)
    dt = time.time() - t1
    print(f"      sv: {len(segs)} 段, 耗时 {dt:.1f}s, RTF={dt/dur:.3f}", flush=True)
    try:
        shutil.rmtree(Path(wav).parent, ignore_errors=True)
    except OSError:
        pass

    SV_CACHE.write_text(
        json.dumps(
            {"language": lang, "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segs]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("[3/3] 与缓存 Qwen3 逐段对比...", flush=True)
    q = json.loads(QWEN_CACHE.read_text(encoding="utf-8"))
    qmap = {round(s["start"], 2): s["text"] for s in q["segments"]}
    svmap = {round(s.start, 2): s.text for s in segs}
    common = sorted(set(qmap) & set(svmap))
    only_q = [k for k in qmap if k not in svmap]
    only_sv = [k for k in svmap if k not in qmap]

    rows = []
    for k in common:
        a, b = norm(qmap[k]), norm(svmap[k])
        sim = difflib.SequenceMatcher(None, a, b).ratio()
        sim_np = difflib.SequenceMatcher(None, norm_nopunct(a), norm_nopunct(b)).ratio()
        rows.append((k, sim, sim_np, a, b))

    n_exact = sum(1 for _, s, _, a, b in rows if a == b)
    n_exact_np = sum(1 for _, _, s, a, b in rows if norm_nopunct(a) == norm_nopunct(b))
    mean_sim = sum(r[1] for r in rows) / len(rows) if rows else 0
    mean_np = sum(r[2] for r in rows) / len(rows) if rows else 0
    q_chars = sum(len(norm(qmap[k])) for k in qmap)
    sv_chars = sum(len(norm(svmap[k])) for k in svmap)

    print("\n================ 汇总 ================")
    print(f"音频时长: {dur:.1f}s ({dur/60:.1f} min)")
    print(f"VAD 段总数: {len(common) + len(only_q) + len(only_sv)}  "
          f"(共同 {len(common)}, 仅Qwen3 {len(only_q)}, 仅SenseVoice {len(only_sv)})")
    print(f"SenseVoice 耗时: {dt:.1f}s   RTF={dt/dur:.3f}")
    print(f"精确匹配(含标点): {n_exact}/{len(rows)} = {n_exact/len(rows)*100:.1f}%")
    print(f"精确匹配(去标点): {n_exact_np}/{len(rows)} = {n_exact_np/len(rows)*100:.1f}%")
    print(f"字符相似度: 含标点 {mean_sim:.3f} | 去标点 {mean_np:.3f}")
    print(f"总字符(去空白): Qwen3={q_chars}  SenseVoice={sv_chars}  差={sv_chars-q_chars:+d}")
    if only_q:
        print(f"⚠ 仅 Qwen3 有文本的段: {only_q[:8]}")
    if only_sv:
        print(f"⚠ 仅 SenseVoice 有文本的段: {only_sv[:8]}")

    # 上轮 auto-language 漂移段复查: 短输出/假名/英文等疑似漂移特征
    if SV_AUTO_CACHE.exists():
        auto = json.loads(SV_AUTO_CACHE.read_text(encoding="utf-8"))
        amap = {round(s["start"], 2): s["text"] for s in auto["segments"]}
        drift_like = [
            k for k in amap
            if k in svmap
            and (len(norm(amap[k])) <= 4 or re.search(r"[ぁ-んァ-ン]|Yeah|okay|\bOK\b", amap[k]))
        ]
        if drift_like:
            print("\n================ 上轮漂移段复查 (auto → zh 固定) ================")
            for k in sorted(drift_like)[:12]:
                mm, ss = int(k // 60), int(k % 60)
                print(f"[{mm:02d}:{ss:02d}] auto: {amap[k]!r}")
                print(f"             zh  : {svmap[k]!r}")

    print("\n================ 抽样: 前 8 段 ================")
    for k, sim, _, a, b in rows[:8]:
        mm, ss = int(k // 60), int(k % 60)
        print(f"\n[{mm:02d}:{ss:02d}] sim={sim:.2f}")
        print(f"  Q3: {a[:110]}")
        print(f"  SV: {b[:110]}")

    print("\n================ 抽样: 相似度最低 4 段 ================")
    for k, sim, _, a, b in sorted(rows, key=lambda r: r[1])[:4]:
        mm, ss = int(k // 60), int(k % 60)
        print(f"\n[{mm:02d}:{ss:02d}] sim={sim:.2f}")
        print(f"  Q3: {a[:160]}")
        print(f"  SV: {b[:160]}")


if __name__ == "__main__":
    asyncio.run(main())
