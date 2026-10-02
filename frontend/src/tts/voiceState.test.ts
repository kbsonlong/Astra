import { describe, expect, it } from "vitest";
import { availabilityLabel, sessionSelectableVoices, voiceAvailability } from "./voiceState";
import type { TtsHealth, TtsVoice, VoiceRevision } from "./voice";

const revision: VoiceRevision = {
  revision: 1, lifecycle: "ready", backend_family: "indextts25_mlx", language: "zh",
  duration_s: 8, speech_duration_s: 8, sample_rate: 24000, channels: 1, sample_width_bytes: 2,
  quality_score: null, default_params: {}, consent_status: "confirmed", consent_confirmed_at: null,
  original_filename: "ref.wav", reference_sha256: "sha", created_at: "2026-01-01", audio_url: "/audio",
  actions: { can_play_reference: true, can_preview: true, can_use_in_session: true, reason: "ready" },
};

const health: TtsHealth = {
  ok: true, mode: "router", engines: {
    indextts_mlx: {
      available: true, reason: "", default: false, user_enabled: true, user_reason: "", pending: 0, capabilities: {},
    },
  },
};

describe("voice state", () => {
  it("requires both lifecycle authorization and a user-enabled live engine", () => {
    expect(voiceAvailability(revision, health)).toMatchObject({ canPreview: true, canUseInSession: true });
    const disabled = structuredClone(health);
    disabled.engines!.indextts_mlx.user_enabled = false;
    disabled.engines!.indextts_mlx.user_reason = "disabled by operator";
    expect(voiceAvailability(revision, disabled)).toMatchObject({ canPreview: false, reason: "disabled by operator" });
    expect(availabilityLabel({ canPlayReference: true, canPreview: false, canUseInSession: false, reason: "revoked" })).toBe("已撤销授权");
  });

  it("does not offer archived or unavailable voices to a live session", () => {
    const active: TtsVoice = {
      voice_id: "active", display_name: "可用", status: "active", current_revision: 1,
      created_at: "", updated_at: "", archived_at: null, revoked_at: null, revision,
    };
    const archived = { ...active, voice_id: "archived", status: "archived" };
    expect(sessionSelectableVoices([active, archived], health)).toEqual([active]);
  });

  it("keeps an unvalidated IndexTTS engine out of preview and session choices", () => {
    const unvalidated = structuredClone(health);
    unvalidated.engines!.indextts_mlx.user_enabled = false;
    unvalidated.engines!.indextts_mlx.user_reason = "device_validation_required";
    unvalidated.engines!.indextts_mlx.validated = false;
    expect(voiceAvailability(revision, unvalidated)).toMatchObject({
      canPreview: false,
      canUseInSession: false,
      reason: "device_validation_required",
    });
  });
});
