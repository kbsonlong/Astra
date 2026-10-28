import { afterEach, describe, expect, it, vi } from "vitest";
import { cancelMeeting, startMeeting, transcribe } from "./transcription";

function response(body: unknown, ok = true): Response {
  return { ok, json: async () => body } as Response;
}

afterEach(() => vi.restoreAllMocks());

describe("transcription API", () => {
  it("uploads a file for basic transcription", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ filename: "a.wav", bytes: 3, text: "你好" }));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["abc"], "a.wav", { type: "audio/wav" });

    await expect(transcribe(file)).resolves.toMatchObject({ text: "你好" });
    expect(fetchMock).toHaveBeenCalledWith("/api/transcribe", expect.objectContaining({ method: "POST" }));
  });

  it("builds meeting options and supports cancellation", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(response({ task_id: "task-1", report_path: "report.md" }))
      .mockResolvedValueOnce(response({ status: "stopped" }));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["abc"], "a.wav");

    await startMeeting(file, "周会", "custom", true);
    await cancelMeeting("task/1");
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/meeting/task%2F1", { method: "DELETE" });
  });
});
