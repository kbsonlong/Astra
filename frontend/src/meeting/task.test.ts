import { describe, expect, it } from "vitest";
import { meetingResultFromStatus } from "./task";

describe("meeting task contract", () => {
  it("maps a terminal API snapshot to the page result shape", () => {
    expect(meetingResultFromStatus({
      task_id: "20261004-120000-abcdef12",
      status: "done",
      duration_s: 12.5,
      segments: 3,
      speakers: ["S1"],
      summary_preview: "摘要",
      transcript_preview: "逐字稿",
    }, "meeting.wav", "/fallback/report.md")).toEqual({
      task_id: "20261004-120000-abcdef12",
      filename: "meeting.wav",
      duration_s: 12.5,
      language: "Chinese",
      segments: 3,
      speakers: ["S1"],
      report_path: "/fallback/report.md",
      summary: "摘要",
      timeline_preview: "逐字稿",
      training_artifacts: undefined,
      prompt_template: undefined,
      separation_audio_urls: undefined,
      overlap_detection: undefined,
    });
  });
});
