import { afterEach, describe, expect, it, vi } from "vitest";
import { createSpeaker, listSpeakerSamples } from "./speakers";

function response(body: unknown, ok = true): Response {
  return { ok, json: async () => body } as Response;
}

afterEach(() => vi.restoreAllMocks());

describe("speaker API", () => {
  it("creates a speaker with a JSON body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ speaker_id: "spk-1" }));
    vi.stubGlobal("fetch", fetchMock);

    await createSpeaker("主持人");

    expect(fetchMock).toHaveBeenCalledWith("/api/speakers", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ display_name: "主持人" }),
    }));
  });

  it("returns an empty sample list when the endpoint is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response({}, false)));

    await expect(listSpeakerSamples("spk/1")).resolves.toEqual([]);
  });
});
