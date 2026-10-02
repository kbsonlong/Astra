import type {
  PreviewAudio,
  RevisionListResponse,
  TtsHealth,
  TtsVoice,
  VoiceListFilters,
  VoiceListResponse,
  VoiceRevision,
  VoiceUpload,
} from "./voice";

const VOICES_PATH = "/api/tts/voices";

export class VoiceApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(detail);
    this.name = "VoiceApiError";
  }
}

type RequestOptions = { signal?: AbortSignal };

function url(path: string, query?: URLSearchParams): string {
  const suffix = query && query.size ? `?${query.toString()}` : "";
  return `${path}${suffix}`;
}

async function errorFor(response: Response): Promise<VoiceApiError> {
  let detail = response.statusText || "Request failed";
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const candidate = body.detail;
      if (typeof candidate === "string") detail = candidate;
    }
  } catch {
    // An HTML proxy error must still surface as a typed request failure.
  }
  return new VoiceApiError(response.status, detail);
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  options: RequestOptions = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { credentials: "same-origin", ...init, signal: options.signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new VoiceApiError(0, "network request failed");
  }
  if (!response.ok) throw await errorFor(response);
  return response.json() as Promise<T>;
}

function uploadForm(upload: VoiceUpload): FormData {
  const form = new FormData();
  form.set("file", upload.file);
  if (upload.language) form.set("language", upload.language);
  if (upload.backendFamily) form.set("backend_family", upload.backendFamily);
  if (upload.defaultParams) form.set("default_params", JSON.stringify(upload.defaultParams));
  return form;
}

export function listVoices(filters: VoiceListFilters = {}, options?: RequestOptions): Promise<VoiceListResponse> {
  const query = new URLSearchParams();
  if (filters.status) query.set("status", filters.status);
  if (filters.language) query.set("language", filters.language);
  if (filters.backendFamily) query.set("backend_family", filters.backendFamily);
  if (filters.query) query.set("q", filters.query);
  if (filters.page) query.set("page", String(filters.page));
  if (filters.pageSize) query.set("page_size", String(filters.pageSize));
  return requestJson(url(VOICES_PATH, query), {}, options);
}

export function getVoice(voiceId: string, options?: RequestOptions): Promise<TtsVoice> {
  return requestJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}`, {}, options);
}

export function listRevisions(voiceId: string, options?: RequestOptions): Promise<RevisionListResponse> {
  return requestJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/revisions`, {}, options);
}

export function createVoice(
  displayName: string,
  upload: VoiceUpload,
  options?: RequestOptions,
): Promise<TtsVoice> {
  const form = uploadForm(upload);
  form.set("name", displayName);
  return requestJson(VOICES_PATH, { method: "POST", body: form }, options);
}

export function addRevision(
  voiceId: string,
  upload: VoiceUpload,
  options?: RequestOptions,
): Promise<VoiceRevision> {
  return requestJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/revisions`, {
    method: "POST",
    body: uploadForm(upload),
  }, options);
}

export function updateVoiceName(voiceId: string, displayName: string, options?: RequestOptions): Promise<TtsVoice> {
  return requestJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  }, options);
}

function postJson<T>(path: string, payload: object, options?: RequestOptions): Promise<T> {
  return requestJson(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }, options);
}

export function confirmConsent(voiceId: string, revision: number, note?: string, options?: RequestOptions): Promise<VoiceRevision> {
  return postJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/revisions/${revision}/consent`, {
    confirmation: "本人已授权 Astra 在本机进行语音合成",
    note,
  }, options);
}

export function revokeConsent(voiceId: string, revision: number, note?: string, options?: RequestOptions): Promise<VoiceRevision> {
  return postJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/revisions/${revision}/revoke-consent`, { note }, options);
}

export function activateVoice(voiceId: string, revision?: number, options?: RequestOptions): Promise<TtsVoice> {
  const query = revision ? `?revision=${revision}` : "";
  return postJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/activate${query}`, {}, options);
}

export function archiveVoice(voiceId: string, options?: RequestOptions): Promise<TtsVoice> {
  return postJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/archive`, {}, options);
}

export function restoreVoice(voiceId: string, options?: RequestOptions): Promise<TtsVoice> {
  return postJson(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/restore`, {}, options);
}

export async function previewVoice(
  voiceId: string,
  revision: number,
  text: string,
  options: RequestOptions = {},
): Promise<PreviewAudio> {
  let response: Response;
  try {
    response = await fetch(`${VOICES_PATH}/${encodeURIComponent(voiceId)}/test`, {
      method: "POST",
      credentials: "same-origin",
      signal: options.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ revision, text }),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new VoiceApiError(0, "network request failed");
  }
  if (!response.ok) throw await errorFor(response);
  const value = (header: string) => response.headers.get(header);
  const parsedRevision = Number(value("X-Astra-Voice-Revision"));
  return {
    audio: await response.blob(),
    contentType: response.headers.get("Content-Type") || "audio/wav",
    voiceId: value("X-Astra-Voice-Id"),
    revision: Number.isSafeInteger(parsedRevision) ? parsedRevision : null,
    referenceSha256: value("X-Astra-Reference-SHA256"),
    backend: value("X-Astra-Backend"),
    modelRevision: value("X-Astra-Model-Revision"),
  };
}

export async function getTtsHealth(options?: RequestOptions): Promise<TtsHealth> {
  const payload = await requestJson<{ tts: TtsHealth }>("/api/health", {}, options);
  return payload.tts;
}

/**
 * Owns one preview surface (for example, a detail drawer). Starting another
 * request cancels its predecessor, so a late response cannot replace the
 * audio selected most recently. The small delay also collapses rapid edits to
 * the preview text into one request.
 */
export class VoicePreviewController {
  private abortController: AbortController | undefined;
  private delayTimer: ReturnType<typeof setTimeout> | undefined;
  private pendingReject: ((reason: unknown) => void) | undefined;

  constructor(private readonly debounceMs = 250) {}

  cancel(): void {
    if (this.delayTimer) clearTimeout(this.delayTimer);
    this.delayTimer = undefined;
    this.abortController?.abort();
    this.abortController = undefined;
    this.pendingReject?.(new DOMException("Preview request was cancelled", "AbortError"));
    this.pendingReject = undefined;
  }

  request(voiceId: string, revision: number, text: string): Promise<PreviewAudio> {
    this.cancel();
    const abortController = new AbortController();
    this.abortController = abortController;
    return new Promise((resolve, reject) => {
      this.pendingReject = reject;
      this.delayTimer = setTimeout(() => {
        this.delayTimer = undefined;
        previewVoice(voiceId, revision, text, { signal: abortController.signal }).then(
          (preview) => {
            if (this.abortController === abortController) {
              this.pendingReject = undefined;
              resolve(preview);
            }
          },
          (error: unknown) => {
            if (this.abortController === abortController) {
              this.pendingReject = undefined;
              reject(error);
            }
          },
        );
      }, this.debounceMs);
    });
  }
}

export function voiceErrorMessage(error: unknown): string {
  if (!(error instanceof VoiceApiError)) return "操作未完成，请稍后重试。";
  switch (error.status) {
    case 0: return "网络连接失败，请检查服务是否可用。";
    case 401: return "登录状态已失效，请重新登录。";
    case 404: return "该音色或版本已不存在，请刷新列表。";
    case 409: return "音色状态已变化，请刷新后再操作。";
    case 413: return "参考音频过大，请选择更小的 WAV 文件。";
    case 415: return "仅支持 WAV 格式的参考音频。";
    case 422: return "提交内容不符合要求，请检查音频、授权或试听文本。";
    case 429: return "当前试听任务较多，请稍后再试。";
    case 503: return "所需语音引擎暂不可用，请稍后再试。";
    default: return "操作未完成，请稍后重试。";
  }
}
