import type { TtsEngineCapability, TtsHealth, TtsVoice, VoiceRevision } from "./voice";

const ENGINE_BY_BACKEND: Record<string, string> = {
  indextts25_mlx: "indextts_mlx",
  indextts_mlx: "indextts_mlx",
  cosyvoice: "cosyvoice",
};

export type VoiceAvailability = {
  canPlayReference: boolean;
  canPreview: boolean;
  canUseInSession: boolean;
  reason: string;
};

export function engineKeyFor(backendFamily: string): string {
  return ENGINE_BY_BACKEND[backendFamily] || backendFamily;
}

export function engineForRevision(revision: VoiceRevision, health?: TtsHealth): TtsEngineCapability | undefined {
  return health?.engines?.[engineKeyFor(revision.backend_family)];
}

export function voiceAvailability(revision: VoiceRevision, health?: TtsHealth): VoiceAvailability {
  const actions = revision.actions;
  if (!actions.can_play_reference) {
    return { canPlayReference: false, canPreview: false, canUseInSession: false, reason: actions.reason };
  }
  if (!actions.can_preview && !actions.can_use_in_session) {
    return {
      canPlayReference: true,
      canPreview: false,
      canUseInSession: false,
      reason: actions.reason,
    };
  }
  const engine = engineForRevision(revision, health);
  if (!engine) {
    return { canPlayReference: true, canPreview: false, canUseInSession: false, reason: "engine_status_unknown" };
  }
  if (!engine.user_enabled) {
    return { canPlayReference: true, canPreview: false, canUseInSession: false, reason: engine.user_reason || "engine_disabled" };
  }
  if (!engine.available) {
    return { canPlayReference: true, canPreview: false, canUseInSession: false, reason: engine.reason || "engine_unavailable" };
  }
  return {
    canPlayReference: true,
    canPreview: actions.can_preview,
    canUseInSession: actions.can_use_in_session,
    reason: actions.reason,
  };
}

export function availabilityLabel(availability: VoiceAvailability): string {
  switch (availability.reason) {
    case "ready": return "可用";
    case "revoked": return "已撤销授权";
    case "consent_required": return "待确认授权";
    case "voice_not_active": return "尚未启用";
    case "revision_not_ready": return "版本尚未就绪";
    case "engine_status_unknown": return "正在检查引擎状态";
    case "engine_disabled": return "引擎未启用";
    case "device_validation_required": return "等待设备验证";
    case "engine_unavailable": return "引擎不可用";
    default: return availability.canUseInSession ? "可用" : "当前不可用";
  }
}

/** Filters list payloads into the choices that are safe to offer a live session. */
export function sessionSelectableVoices(voices: TtsVoice[], health?: TtsHealth): TtsVoice[] {
  return voices.filter((voice) => {
    if (voice.status !== "active" || !voice.revision) return false;
    return voiceAvailability(voice.revision, health).canUseInSession;
  });
}
