import { describe, expect, it } from "vitest";
import {
  acceptsGenerationEvent,
  formatElapsedTime,
  generationForNewConnection,
  heartbeatExpired,
  nextActiveGeneration,
  ReconnectPcmBuffer,
  startSessionPayload,
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

  it("pins the selected voice revision in the session handshake", () => {
    expect(startSessionPayload("recording-1", { voiceId: "voice-1", revision: 3 })).toEqual({
      type: "start_session",
      recording_id: "recording-1",
      voice_id: "voice-1",
      voice_revision: 3,
    });
    expect(startSessionPayload(null, null)).toEqual({ type: "start_session" });
  });
});

describe("long-running session reliability", () => {
  it("keeps a bounded reconnect buffer and reports dropped audio", () => {
    const buffer = new ReconnectPcmBuffer(10, 0.03);
    const first = new ArrayBuffer(1);
    const second = new ArrayBuffer(2);
    const third = new ArrayBuffer(3);
    const fourth = new ArrayBuffer(4);

    expect(buffer.push(first)).toBe(0);
    expect(buffer.push(second)).toBe(0);
    expect(buffer.push(third)).toBe(0);
    expect(buffer.push(fourth)).toBeCloseTo(0.01);
    expect(buffer.durationSeconds).toBeCloseTo(0.03);
    expect(buffer.drain()).toEqual([second, third, fourth]);
    expect(buffer.length).toBe(0);
  });

  it("detects a silent server and formats multi-hour recording durations", () => {
    expect(heartbeatExpired(1_000, 45_999)).toBe(false);
    expect(heartbeatExpired(1_000, 46_000)).toBe(true);
    expect(formatElapsedTime(3_661.9)).toBe("01:01:01");
  });
});
