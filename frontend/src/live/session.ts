export type GenerationEvent = {
  type: string;
  generation_id?: number;
};

export type VoiceSnapshot = { voiceId: string; revision: number };

/** Creates the immutable voice selection sent when a WebSocket Session begins. */
export function startSessionPayload(recordingId: string | null, voice: VoiceSnapshot | null): {
  type: "start_session";
  recording_id?: string;
  voice_id?: string;
  voice_revision?: number;
} {
  return {
    type: "start_session",
    ...(recordingId ? { recording_id: recordingId } : {}),
    ...(voice ? { voice_id: voice.voiceId, voice_revision: voice.revision } : {}),
  };
}

export const HEARTBEAT_INTERVAL_MS = 15_000;
export const HEARTBEAT_TIMEOUT_MS = 45_000;
export const RECONNECT_AUDIO_BUFFER_SECONDS = 30;

/** Keep a short PCM bridge across reconnects without allowing browser memory to grow forever. */
export class ReconnectPcmBuffer {
  private readonly framesPerSecond: number;
  private readonly maxFrames: number;
  private frames: ArrayBuffer[] = [];
  private head = 0;

  constructor(frameDurationMs: number, maxSeconds = RECONNECT_AUDIO_BUFFER_SECONDS) {
    if (frameDurationMs <= 0 || maxSeconds <= 0) {
      throw new Error("PCM reconnect buffer limits must be positive");
    }
    this.framesPerSecond = 1000 / frameDurationMs;
    this.maxFrames = Math.max(1, Math.floor(this.framesPerSecond * maxSeconds));
  }

  get length(): number {
    return this.frames.length - this.head;
  }

  get durationSeconds(): number {
    return this.length / this.framesPerSecond;
  }

  /** Returns the duration dropped from the oldest edge of the queue. */
  push(frame: ArrayBuffer): number {
    this.frames.push(frame);
    if (this.length <= this.maxFrames) return 0;
    this.head += 1;
    this.compactIfNeeded();
    return 1 / this.framesPerSecond;
  }

  drain(): ArrayBuffer[] {
    const pending = this.frames.slice(this.head);
    this.clear();
    return pending;
  }

  clear(): void {
    this.frames = [];
    this.head = 0;
  }

  private compactIfNeeded(): void {
    if (this.head >= 1024 && this.head * 2 >= this.frames.length) {
      this.frames = this.frames.slice(this.head);
      this.head = 0;
    }
  }
}

export function heartbeatExpired(
  lastServerMessageAt: number,
  now: number,
  timeoutMs = HEARTBEAT_TIMEOUT_MS,
): boolean {
  return now - lastServerMessageAt >= timeoutMs;
}

export function formatElapsedTime(totalSeconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const seconds = safeSeconds % 60;
  return [hours, minutes, seconds].map((value) => String(value).padStart(2, "0")).join(":");
}

/** A new WebSocket owns a new server-side Session and generation sequence. */
export function generationForNewConnection(): number {
  return 0;
}

/**
 * Reject only stale generation-scoped payloads. State-change messages are
 * intentionally unscoped so reconnect handshakes can always restore UI state.
 */
export function acceptsGenerationEvent(event: GenerationEvent, activeGeneration: number): boolean {
  return (
    event.type === "state_change" ||
    event.generation_id === undefined ||
    event.generation_id >= activeGeneration
  );
}

export function nextActiveGeneration(event: GenerationEvent, activeGeneration: number): number {
  if (event.type === "state_change" || event.generation_id === undefined) {
    return activeGeneration;
  }
  return Math.max(activeGeneration, event.generation_id);
}
