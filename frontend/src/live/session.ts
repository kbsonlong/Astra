export type GenerationEvent = {
  type: string;
  generation_id?: number;
};

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
