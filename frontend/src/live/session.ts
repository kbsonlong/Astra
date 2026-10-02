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
