import { describe, expect, it } from "vitest";
import {
  acceptsGenerationEvent,
  generationForNewConnection,
  nextActiveGeneration,
} from "./session";

describe("WebSocket generation handling", () => {
  it("accepts the first generation after a reconnect", () => {
    const oldConnectionGeneration = 4;
    const newConnectionGeneration = generationForNewConnection();
    const event = { type: "asr_final", generation_id: 1 };

    expect(acceptsGenerationEvent(event, oldConnectionGeneration)).toBe(false);
    expect(acceptsGenerationEvent(event, newConnectionGeneration)).toBe(true);
    expect(nextActiveGeneration(event, newConnectionGeneration)).toBe(1);
  });

  it("continues rejecting an older payload within one connection", () => {
    expect(acceptsGenerationEvent({ type: "tts_chunk", generation_id: 2 }, 3)).toBe(false);
    expect(acceptsGenerationEvent({ type: "state_change", generation_id: 2 }, 3)).toBe(true);
  });
});
