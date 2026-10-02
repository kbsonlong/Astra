import { afterEach, describe, expect, it, vi } from "vitest";
import { listVoices, previewVoice, VoiceApiError, VoicePreviewController, voiceErrorMessage } from "./voiceApi";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("voice API", () => {
  it("encodes list filters using the backend contract", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ items: [], pagination: {} })));
    globalThis.fetch = fetchMock;

    await listVoices({ status: "active", backendFamily: "indextts25_mlx", query: "王 小" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/tts/voices?status=active&backend_family=indextts25_mlx&q=%E7%8E%8B+%E5%B0%8F",
      expect.objectContaining({ credentials: "same-origin" }),
    );
  });

  it("keeps preview provenance metadata with its audio blob", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(new Response(new Uint8Array([1, 2]), {
      headers: {
        "Content-Type": "audio/wav",
        "X-Astra-Voice-Id": "voice-1",
        "X-Astra-Voice-Revision": "2",
        "X-Astra-Reference-SHA256": "sha",
        "X-Astra-Backend": "indextts25_mlx",
        "X-Astra-Model-Revision": "model-r1",
      },
    }));

    const preview = await previewVoice("voice-1", 2, "你好");

    expect(preview).toMatchObject({ voiceId: "voice-1", revision: 2, backend: "indextts25_mlx", modelRevision: "model-r1" });
    expect(preview.audio.size).toBe(2);
  });

  it("maps a saturated preview queue to a recoverable message", () => {
    expect(voiceErrorMessage(new VoiceApiError(429, "voice preview synthesis failed"))).toContain("试听任务");
  });

  it("cancels a delayed preview when a newer choice replaces it", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockResolvedValue(new Response(new Uint8Array([1])));
    globalThis.fetch = fetchMock;
    const previews = new VoicePreviewController(200);
    const first = previews.request("voice-1", 1, "旧文本");
    const firstRejected = expect(first).rejects.toMatchObject({ name: "AbortError" });
    const second = previews.request("voice-1", 1, "新文本");
    await vi.advanceTimersByTimeAsync(200);
    await firstRejected;
    await expect(second).resolves.toMatchObject({ audio: expect.any(Blob) });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    previews.cancel();
    vi.useRealTimers();
  });
});
