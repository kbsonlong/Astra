export type TranscriptionResult = { filename: string; bytes: number; text: string };
export type TimelineSegment = { index: number; start: number; end: number; text: string; source_index?: number };
export type EnhancementStatus = { stage_name: string; status: "applied" | "not_applicable" | "failed" | "disabled"; latency_ms: number; fallback_reason?: string | null };
export type SeparationStatus = { stage_name: string; status: "applied" | "not_applicable" | "failed" | "disabled"; output_count: number; latency_ms: number; fallback_reason?: string | null };
export type MeetingJob = { task_id: string; report_path?: string };

type StreamHandlers = {
  onResult: (result: TranscriptionResult) => void;
  onCorrection: (token: string) => void;
  onSegment: (segment: TimelineSegment) => void;
  onCorrectionSegment: (index: number, text: string) => void;
  onEnhancement: (stages: EnhancementStatus[]) => void;
  onSeparation: (stage: SeparationStatus) => void;
  onError: (message: string) => void;
};

export async function transcribe(file: File): Promise<TranscriptionResult> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/transcribe", { method: "POST", body: form });
  const payload = await response.json() as TranscriptionResult & { detail?: string };
  if (!response.ok) throw new Error(payload.detail ?? "上传失败");
  return payload;
}

export async function streamTranscription(file: File, separate: boolean, handlers: StreamHandlers): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(separate ? "/api/transcribe/stream?separate=true" : "/api/transcribe/stream", { method: "POST", body: form });
  if (!response.ok) {
    const payload = await response.json() as { detail?: string };
    throw new Error(payload.detail ?? "流式修正失败");
  }
  if (!response.body) throw new Error("服务端未返回流");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const event of events) {
      const line = event.split("\n").find((item) => item.startsWith("data: "));
      if (!line || line.slice(6) === "[DONE]") continue;
      const payload = JSON.parse(line.slice(6)) as {
        type: string; text?: string; token?: string; message?: string; index?: number; start?: number; end?: number; source_index?: number;
        stages?: EnhancementStatus[]; stage?: SeparationStatus;
      };
      if (payload.type === "asr_final") handlers.onResult({ filename: file.name, bytes: file.size, text: payload.text ?? "" });
      if (payload.type === "enhancement_status") handlers.onEnhancement(payload.stages ?? []);
      if (payload.type === "separation_status" && payload.stage) handlers.onSeparation(payload.stage);
      if (payload.type === "asr_segment" && payload.index !== undefined) handlers.onSegment({ index: payload.index, start: payload.start ?? 0, end: payload.end ?? 0, text: payload.text ?? "", source_index: payload.source_index });
      if (payload.type === "correction_token") handlers.onCorrection(payload.token ?? "");
      if (payload.type === "correction_segment" && payload.index !== undefined) handlers.onCorrectionSegment(payload.index, payload.text ?? "");
      if (payload.type === "error") handlers.onError(payload.message ?? "流式修正失败，已保留原始文本");
    }
    if (done) break;
  }
}

export async function startMeeting(file: File, topic: string, promptTemplate: string, separate: boolean): Promise<MeetingJob> {
  const form = new FormData();
  form.append("file", file);
  form.append("topic", topic);
  form.append("prompt_template", promptTemplate);
  if (separate) form.append("separate", "true");
  const response = await fetch("/api/meeting/process", { method: "POST", body: form });
  const payload = await response.json() as MeetingJob & { detail?: string };
  if (!response.ok) throw new Error(payload.detail ?? "会议处理失败");
  return payload;
}

export async function cancelMeeting(taskId: string): Promise<{ status?: string }> {
  const response = await fetch(`/api/meeting/${encodeURIComponent(taskId)}`, { method: "DELETE" });
  const payload = await response.json() as { detail?: string; status?: string };
  if (!response.ok) throw new Error(payload.detail ?? "取消会议处理失败");
  return payload;
}
