export type VoiceStatus = "draft" | "active" | "archived" | "revoked" | string;
export type RevisionLifecycle = "draft" | "ready" | "revoked" | string;
export type ConsentStatus = "pending" | "confirmed" | "revoked" | string;

export type VoiceActions = {
  can_play_reference: boolean;
  can_preview: boolean;
  can_use_in_session: boolean;
  reason: string;
};

export type VoiceRevision = {
  revision: number;
  lifecycle: RevisionLifecycle;
  backend_family: string;
  language: string;
  duration_s: number;
  speech_duration_s: number;
  sample_rate: number;
  channels: number;
  sample_width_bytes: number;
  quality_score: number | null;
  default_params: Record<string, unknown>;
  consent_status: ConsentStatus;
  consent_confirmed_at: string | null;
  original_filename: string;
  reference_sha256: string;
  created_at: string;
  audio_url: string;
  actions: VoiceActions;
};

export type TtsVoice = {
  voice_id: string;
  display_name: string;
  status: VoiceStatus;
  current_revision: number | null;
  created_at: string;
  updated_at: string;
  archived_at: string | null;
  revoked_at: string | null;
  revision: VoiceRevision | null;
};

export type VoicePagination = {
  page: number;
  page_size: number;
  total: number;
  pages: number;
};

export type VoiceListResponse = {
  items: TtsVoice[];
  pagination: VoicePagination;
};

export type RevisionListResponse = { items: VoiceRevision[] };

export type TtsEngineCapability = {
  available: boolean;
  reason: string;
  default: boolean;
  user_enabled: boolean;
  user_reason: string;
  validated?: boolean;
  pending: number;
  capabilities: Record<string, unknown>;
};

export type TtsHealth = {
  ok: boolean;
  mode: string;
  available?: boolean;
  reason?: string;
  default_backend?: string;
  fallback_backend?: string | null;
  engines?: Record<string, TtsEngineCapability>;
  capabilities?: Record<string, unknown>;
};

export type VoiceListFilters = {
  status?: string;
  language?: string;
  backendFamily?: string;
  query?: string;
  page?: number;
  pageSize?: number;
};

export type VoiceUpload = {
  file: File;
  language?: string;
  backendFamily?: string;
  defaultParams?: Record<string, unknown>;
};

export type PreviewAudio = {
  audio: Blob;
  contentType: string;
  voiceId: string | null;
  revision: number | null;
  referenceSha256: string | null;
  backend: string | null;
  modelRevision: string | null;
};
