"""Build a focused prompt for LLM-generated meeting brief artwork."""
from __future__ import annotations

from pathlib import Path


def build_report_image_prompt(report_path: str | Path) -> str:
    """Turn the generated report into a concise, visual image brief."""
    report = Path(report_path).read_text(encoding="utf-8")
    summary = report.split("## 🌐", 1)[0].split("## 🎙️", 1)[0]
    summary = summary.replace("```", " ")
    summary = "\n".join(line.strip() for line in summary.splitlines() if line.strip())[:6000]
    return (
        "Create a polished editorial infographic-style meeting brief image based only on the "
        "following Chinese meeting summary. Use a calm professional blue and violet palette, "
        "clear visual hierarchy, subtle icons for conclusions, action items, and open questions, "
        "and a clean modern business presentation style. Preserve the meaning and priority of "
        "the content. Do not invent people, numbers, logos, or facts. Do not include a transcript. "
        "If text is rendered, use short legible Simplified Chinese labels; never use gibberish text.\n\n"
        f"Meeting summary:\n{summary}"
    )
