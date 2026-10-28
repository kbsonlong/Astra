export type SpeakerProfile = {
  speaker_id: string;
  display_name: string;
  status: string;
  sample_count: number;
  embedding_model: string;
  updated_at: string;
};

export type SpeakerSample = {
  sample_id: string;
  duration_s: number;
  speech_duration_s: number;
  quality_score: number | null;
  original_filename: string;
  audio_url: string | null;
  created_at: string;
};

async function json<T>(response: Response): Promise<T> {
  return await response.json() as T;
}

export async function listSpeakers(): Promise<SpeakerProfile[]> {
  const response = await fetch("/api/speakers");
  const payload = await json<{ items?: SpeakerProfile[]; detail?: string }>(response);
  if (!response.ok) throw new Error(payload.detail ?? "无法读取声纹档案");
  return payload.items ?? [];
}

export async function listSpeakerSamples(id: string): Promise<SpeakerSample[]> {
  const response = await fetch(`/api/speakers/${encodeURIComponent(id)}/samples`);
  if (!response.ok) return [];
  const payload = await json<{ items?: SpeakerSample[] }>(response);
  return payload.items ?? [];
}

async function mutateSpeaker(id: string, init: RequestInit, fallback: string): Promise<SpeakerProfile> {
  const response = await fetch(`/api/speakers/${encodeURIComponent(id)}`, init);
  const payload = await json<SpeakerProfile & { detail?: string }>(response);
  if (!response.ok) throw new Error(payload.detail ?? fallback);
  return payload;
}

export async function createSpeaker(displayName: string): Promise<SpeakerProfile> {
  const response = await fetch("/api/speakers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  });
  const payload = await json<SpeakerProfile & { detail?: string }>(response);
  if (!response.ok) throw new Error(payload.detail ?? "创建失败");
  return payload;
}

export function renameSpeaker(id: string, displayName: string): Promise<SpeakerProfile> {
  return mutateSpeaker(id, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  }, "改名失败");
}

export async function approveSpeaker(id: string): Promise<SpeakerProfile> {
  const response = await fetch(`/api/speakers/${encodeURIComponent(id)}/approve`, { method: "POST" });
  const payload = await json<SpeakerProfile & { detail?: string }>(response);
  if (!response.ok) throw new Error(payload.detail ?? "审核失败");
  return payload;
}

export async function disableSpeaker(id: string): Promise<SpeakerProfile> {
  const response = await fetch(`/api/speakers/${encodeURIComponent(id)}`, { method: "DELETE" });
  const payload = await json<SpeakerProfile & { detail?: string }>(response);
  if (!response.ok) throw new Error(payload.detail ?? "禁用失败");
  return payload;
}

export async function uploadSpeakerSample(id: string, file: File): Promise<{ speech_duration_s?: number; quality_score?: number }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`/api/speakers/${encodeURIComponent(id)}/samples`, { method: "POST", body: form });
  const payload = await json<{ detail?: string; speech_duration_s?: number; quality_score?: number }>(response);
  if (!response.ok) throw new Error(payload.detail ?? "样本注册失败");
  return payload;
}
