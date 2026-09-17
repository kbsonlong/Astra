import { ChangeEvent, FormEvent, Fragment, ReactNode, useEffect, useRef, useState } from "react";

type TranscriptionResult = {
  filename: string;
  bytes: number;
  text: string;
};


type TimelineSegment = {
  index: number;
  start: number;
  end: number;
  text: string;
  source_index?: number;
};

type EnhancementStatus = {
  stage_name: string;
  status: "applied" | "not_applicable" | "failed" | "disabled";
  latency_ms: number;
  fallback_reason?: string | null;
};

type SeparationStatus = {
  stage_name: string;
  status: "applied" | "not_applicable" | "failed" | "disabled";
  output_count: number;
  latency_ms: number;
  fallback_reason?: string | null;
};

type MeetingPromptTemplate = {
  id: string;
  name: string;
  description: string;
  chunk_system_prompt: string;
  merge_system_prompt: string;
  builtin?: boolean;
  editable?: boolean;
};

type PromptEditorState = {
  id?: string;
  name: string;
  description: string;
  chunk_system_prompt: string;
  merge_system_prompt: string;
};

type OverlapDetection = {
  status?: "ok" | "failed";
  suspected: boolean;
  score: number;
  active_frames: number;
  candidate_frames: number;
  threshold: number;
  details?: {
    sample_rate?: number;
    frame_seconds?: number;
    hop_seconds?: number;
    frequency_band_hz?: number[];
    eligible_frames?: number;
  };
};

function OverlapDetectionView({ detection }: { detection: OverlapDetection }) {
  const score = Math.max(0, Math.min(1, detection.score));
  const threshold = Math.max(0, Math.min(1, detection.threshold));
  const details = detection.details ?? {};
  const eligibleFrames = details.eligible_frames ?? detection.active_frames;
  const candidateRatio = eligibleFrames > 0
    ? detection.candidate_frames / eligibleFrames
    : 0;
  const band = details.frequency_band_hz?.join("–") ?? "80–350";
  return (
    <div className={"overlap-detection " + (detection.suspected ? "is-suspected" : "is-clear")}>
      <div className="overlap-detection-head">
        <strong>重叠检测</strong>
        <span>
          {detection.status === "failed"
            ? "专用检测器不可用，未触发分离"
            : detection.suspected
              ? "疑似重叠，已触发分离"
              : "未发现明显重叠"}
        </span>
      </div>
      <div className="overlap-meter-wrap">
        <div
          className="overlap-meter"
          role="progressbar"
          aria-label="重叠检测得分"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(score * 100)}
        >
          <span style={{ width: (score * 100) + "%" }} />
          <i style={{ left: (threshold * 100) + "%" }} />
        </div>
        <div className="overlap-meter-labels">
          <span>得分 {(score * 100).toFixed(1)}%</span>
          <span>触发线 {(threshold * 100).toFixed(1)}%</span>
        </div>
      </div>
      <div className="overlap-detection-stats">
        <span>候选帧 {(candidateRatio * 100).toFixed(1)}%</span>
        <span>有效帧 {detection.active_frames}</span>
        <span>分析频段 {band} Hz</span>
      </div>
      <small>
        {details.sample_rate ?? 16000} Hz · 窗长 {((details.frame_seconds ?? 0.05) * 1000).toFixed(0)} ms ·
        步长 {((details.hop_seconds ?? 0.01) * 1000).toFixed(0)} ms
      </small>
    </div>
  );
}

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function renderInlineMarkdown(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}

function MarkdownContent({ source }: { source: string }) {
  const lines = source.split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trimEnd();
    if (!line.trim()) {
      index += 1;
      continue;
    }
    if (line.startsWith("```") || line.startsWith("~~~")) {
      const fence = line.slice(0, 3);
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].startsWith(fence)) {
        code.push(lines[index]);
        index += 1;
      }
      index += 1;
      blocks.push(<pre key={blocks.length}><code>{code.join("\n")}</code></pre>);
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const Tag = `h${heading[1].length}` as "h1" | "h2" | "h3";
      blocks.push(<Tag key={blocks.length}>{renderInlineMarkdown(heading[2])}</Tag>);
      index += 1;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index])) {
        items.push(<li key={items.length}>{renderInlineMarkdown(lines[index].replace(/^[-*]\s+/, ""))}</li>);
        index += 1;
      }
      blocks.push(<ul key={blocks.length}>{items}</ul>);
      continue;
    }
    const paragraph: string[] = [line];
    index += 1;
    while (index < lines.length && lines[index].trim() && !/^(#{1,3})\s|^[-*]\s+|^```|^~~~/.test(lines[index])) {
      paragraph.push(lines[index].trimEnd());
      index += 1;
    }
    blocks.push(<p key={blocks.length}>{renderInlineMarkdown(paragraph.join(" "))}</p>);
  }
  return <div className="markdown-content">{blocks}</div>;
}

type MeetingResult = {
  task_id: string;
  filename: string;
  duration_s: number;
  language: string;
  segments: number;
  speakers: string[];
  report_path: string;
  summary: string;
  timeline_preview: string;
  prompt_template?: string;
  training_artifacts?: {
    clips_dir?: string;
    transcript_segments?: string;
    training_candidates?: string;
    candidate_segments?: number;
  };
  separation_audio_urls?: string[];
  overlap_detection?: OverlapDetection;
};

type MeetingStatus = {
  type?: string;
  task_id: string;
  status: "processing" | "done" | "failed" | "stopped";
  duration_s?: number;
  segments?: number;
  speakers?: string[];
  report_path?: string;
  summary_preview?: string;
  transcript_preview?: string;
  elapsed_s?: number;
  error?: string;
  message?: string;
  training_artifacts?: MeetingResult["training_artifacts"];
  prompt_template?: string;
  separation_audio_urls?: string[];
  overlap_detection?: OverlapDetection;
};

const meetingEventUrl = (taskId: string) => {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/api/meeting/${encodeURIComponent(taskId)}/events`;
};

const notificationEventUrl = () => {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/api/notifications/events`;
};

type SpeakerProfile = {
  speaker_id: string;
  display_name: string;
  status: string;
  sample_count: number;
  embedding_model: string;
  updated_at: string;
};

type SpeakerNotification = {
  id: string;
  message: string;
  speaker_id: string;
};

type NotificationEvent = {
  type: "notifications";
  items?: SpeakerNotification[];
};

type SpeakerSample = {
  sample_id: string;
  duration_s: number;
  speech_duration_s: number;
  quality_score: number | null;
  original_filename: string;
  audio_url: string | null;
  created_at: string;
};

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [correction, setCorrection] = useState("");
  const [correctionTimeline, setCorrectionTimeline] = useState<TimelineSegment[]>([]);
  const [enhancementStatus, setEnhancementStatus] = useState<EnhancementStatus[]>([]);
  const [separationStatus, setSeparationStatus] = useState<SeparationStatus | null>(null);
  const [status, setStatus] = useState("选择一个音频文件开始测试");
  const [busy, setBusy] = useState(false);
  const [topic, setTopic] = useState("");
  const [streamSeparate, setStreamSeparate] = useState(false);
  const [separateSpeakers, setSeparateSpeakers] = useState(false);
  const [promptTemplates, setPromptTemplates] = useState<MeetingPromptTemplate[]>([]);
  const [promptTemplate, setPromptTemplate] = useState("standard");
  const [promptEditor, setPromptEditor] = useState<PromptEditorState | null>(null);
  const [promptEditorStatus, setPromptEditorStatus] = useState("");
  const [meeting, setMeeting] = useState<MeetingResult | null>(null);
  const [activeMeetingTask, setActiveMeetingTask] = useState<string | null>(null);
  const [speakers, setSpeakers] = useState<SpeakerProfile[]>([]);
  const [speakerName, setSpeakerName] = useState("");
  const [speakerStatus, setSpeakerStatus] = useState("声纹档案未加载");
  const [renaming, setRenaming] = useState<Record<string, string>>({});
  const [speakerNotifications, setSpeakerNotifications] = useState<SpeakerNotification[]>([]);
  const [speakerSamples, setSpeakerSamples] = useState<Record<string, SpeakerSample[]>>({});
  const meetingSocket = useRef<WebSocket | null>(null);
  const cancelledMeetingTask = useRef<string | null>(null);
  const notificationSocket = useRef<WebSocket | null>(null);

  useEffect(() => {
    void refreshSpeakers();
    void loadPromptTemplates();
    connectNotifications();
    return () => {
      meetingSocket.current?.close();
      notificationSocket.current?.close();
    };
  }, []);

  async function loadPromptTemplates(preferredId?: string) {
    try {
      const response = await fetch("/api/meeting/prompt-templates");
      const payload = await response.json() as { default?: string; items?: MeetingPromptTemplate[] };
      if (!response.ok) return;
      const items = payload.items ?? [];
      setPromptTemplates(items);
      setPromptTemplate((current) => {
        if (preferredId && items.some((item) => item.id === preferredId)) return preferredId;
        if (items.some((item) => item.id === current)) return current;
        return payload.default ?? items[0]?.id ?? "standard";
      });
    } catch {
      // The standard template remains usable if metadata loading fails.
    }
  }

  function openNewPromptTemplate() {
    const selected = promptTemplates.find((template) => template.id === promptTemplate);
    setPromptEditor({
      name: selected ? `${selected.name}（自定义）` : "我的会议纪要",
      description: selected?.description ?? "",
      chunk_system_prompt: selected?.chunk_system_prompt ?? "你是严谨的会议纪要助手。只根据逐字稿提取明确说出的信息，不补充或臆造事实。输出简体中文 markdown。",
      merge_system_prompt: selected?.merge_system_prompt ?? "你是会议纪要总编。合并分段要点，去除重复内容，不新增逐字稿中没有的事实。输出简体中文 markdown。",
    });
    setPromptEditorStatus("");
  }

  function openEditPromptTemplate() {
    const selected = promptTemplates.find((template) => template.id === promptTemplate);
    if (!selected?.editable) return;
    setPromptEditor({
      id: selected.id,
      name: selected.name,
      description: selected.description,
      chunk_system_prompt: selected.chunk_system_prompt,
      merge_system_prompt: selected.merge_system_prompt,
    });
    setPromptEditorStatus("");
  }

  function updatePromptEditor(field: keyof Omit<PromptEditorState, "id">, value: string) {
    setPromptEditor((current) => current ? { ...current, [field]: value } : current);
  }

  async function savePromptTemplate() {
    if (!promptEditor) return;
    setPromptEditorStatus("保存中...");
    try {
      const response = await fetch(
        promptEditor.id
          ? `/api/meeting/prompt-templates/${encodeURIComponent(promptEditor.id)}`
          : "/api/meeting/prompt-templates",
        {
          method: promptEditor.id ? "PUT" : "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: promptEditor.name,
            description: promptEditor.description,
            chunk_system_prompt: promptEditor.chunk_system_prompt,
            merge_system_prompt: promptEditor.merge_system_prompt,
          }),
        },
      );
      const payload = await response.json() as { detail?: string; item?: MeetingPromptTemplate };
      if (!response.ok || !payload.item) throw new Error(payload.detail ?? "模板保存失败");
      setPromptEditor(null);
      setPromptEditorStatus("");
      await loadPromptTemplates(payload.item.id);
      setStatus("纪要模板已保存");
    } catch (error) {
      setPromptEditorStatus(error instanceof Error ? error.message : "模板保存失败");
    }
  }

  async function deletePromptTemplate() {
    const selected = promptTemplates.find((template) => template.id === promptTemplate);
    if (!selected?.editable || !window.confirm(`确定删除“${selected.name}”吗？`)) return;
    try {
      const response = await fetch(`/api/meeting/prompt-templates/${encodeURIComponent(selected.id)}`, { method: "DELETE" });
      const payload = await response.json() as { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "模板删除失败");
      await loadPromptTemplates("standard");
      setPromptEditorStatus("");
      setStatus("自定义纪要模板已删除");
    } catch (error) {
      setPromptEditorStatus(error instanceof Error ? error.message : "模板删除失败");
    }
  }

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setResult(null);
    setCorrection("");
    setCorrectionTimeline([]);
    setEnhancementStatus([]);
    setSeparationStatus(null);
    setMeeting(null);
    setActiveMeetingTask(null);
    cancelledMeetingTask.current = null;
    meetingSocket.current?.close();
    meetingSocket.current = null;
    setStatus(event.target.files?.[0]?.name ?? "选择一个音频文件开始测试");
  }

  function waitForMeeting(taskId: string, reportPath: string, filename: string): Promise<MeetingResult> {
    meetingSocket.current?.close();
    return new Promise((resolve, reject) => {
      const connection = new WebSocket(meetingEventUrl(taskId));
      let finished = false;
      meetingSocket.current = connection;

      connection.onmessage = (message) => {
        let current: MeetingStatus;
        try {
          current = JSON.parse(message.data) as MeetingStatus;
        } catch {
          finished = true;
          connection.close();
          reject(new Error("会议状态 WebSocket 返回了无法识别的消息"));
          return;
        }
        if (current.type === "error") {
          finished = true;
          connection.close();
          reject(new Error(current.message ?? "会议状态订阅失败"));
          return;
        }
        if (current.status === "processing") {
          setStatus(`会议处理中 ${current.elapsed_s ?? 0}s`);
          return;
        }
        if (current.status === "failed" || current.status === "stopped") {
          finished = true;
          connection.close();
          reject(new Error(current.error ?? (current.status === "stopped" ? "会议已取消" : "会议处理失败，请查看服务端日志")));
          return;
        }
        finished = true;
        resolve({
          task_id: current.task_id,
          filename,
          duration_s: current.duration_s ?? 0,
          language: "Chinese",
          segments: current.segments ?? 0,
          speakers: current.speakers ?? [],
          report_path: current.report_path ?? reportPath,
          summary: current.summary_preview ?? "",
          timeline_preview: current.transcript_preview ?? "",
          training_artifacts: current.training_artifacts,
          prompt_template: current.prompt_template,
          separation_audio_urls: current.separation_audio_urls,
          overlap_detection: current.overlap_detection,
        });
      };

      connection.onerror = () => {
        if (!finished) {
          finished = true;
          reject(new Error("会议状态 WebSocket 连接失败"));
        }
      };
      connection.onclose = () => {
        if (meetingSocket.current === connection) meetingSocket.current = null;
        if (!finished) {
          finished = true;
          reject(new Error(
            cancelledMeetingTask.current === taskId ? "会议已取消" : "会议状态 WebSocket 已断开",
          ));
        }
      };
    });
  }

  async function uploadWithCorrection() {
    if (!file) {
      setStatus("请先选择音频文件");
      return;
    }
    setBusy(true);
    setResult(null);
    setCorrection("");
    setCorrectionTimeline([]);
    setEnhancementStatus([]);
    setSeparationStatus(null);
    setStatus("正在转写并流式修正...");
    const form = new FormData();
    form.append("file", file);
    try {
      const streamPath = streamSeparate
        ? "/api/transcribe/stream?separate=true"
        : "/api/transcribe/stream";
      const response = await fetch(streamPath, { method: "POST", body: form });
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
          if (!line) continue;
          const data = line.slice(6);
          if (data === "[DONE]") continue;
          const payload = JSON.parse(data) as {
            type: string; text?: string; token?: string; message?: string;
            index?: number; start?: number; end?: number; source_index?: number;
            stages?: EnhancementStatus[];
            stage?: SeparationStatus;
          };
          if (payload.type === "asr_final") setResult({ filename: file.name, bytes: file.size, text: payload.text ?? "" });
          if (payload.type === "enhancement_status") setEnhancementStatus(payload.stages ?? []);
          if (payload.type === "separation_status" && payload.stage) setSeparationStatus(payload.stage);
          if (payload.type === "asr_segment" && payload.index !== undefined) {
            const segment = {
              index: payload.index,
              start: payload.start ?? 0,
              end: payload.end ?? 0,
              text: payload.text ?? "",
              source_index: payload.source_index,
            };
            setCorrectionTimeline((current) => [...current.filter((item) => item.index !== segment.index), segment].sort((a, b) => a.start - b.start));
          }
          if (payload.type === "correction_token") setCorrection((current) => current + (payload.token ?? ""));
          if (payload.type === "correction_segment" && payload.index !== undefined) {
            setCorrectionTimeline((current) => current.map((item) => item.index === payload.index ? { ...item, text: payload.text ?? item.text } : item));
          }
          if (payload.type === "error") setStatus(payload.message ?? "流式修正失败，已保留原始文本");
        }
        if (done) break;
      }
      setStatus("流式修正完成");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "流式修正失败");
    } finally {
      setBusy(false);
    }
  }

  async function upload(event: FormEvent) {
    event.preventDefault();
    if (!file) {
      setStatus("请先选择音频文件");
      return;
    }
    setBusy(true);
    setResult(null);
    setStatus("正在转写...");
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/transcribe", { method: "POST", body: form });
      const payload = await response.json() as TranscriptionResult | { detail?: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "上传失败");
      setResult(payload as TranscriptionResult);
      setStatus("转写完成");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "上传失败");
    } finally {
      setBusy(false);
    }
  }

  async function runMeeting() {
    if (!file) {
      setStatus("请先选择音频文件");
      return;
    }
    setBusy(true);
    setMeeting(null);
    setStatus("正在处理会议(转写→说话人分离→AI纪要)，几分钟，请稍候...");
    const form = new FormData();
    form.append("file", file);
    form.append("topic", topic);
    form.append("prompt_template", promptTemplate);
    if (separateSpeakers) form.append("separate", "true");
    let taskId: string | null = null;
    try {
      const response = await fetch("/api/meeting/process", { method: "POST", body: form });
      const payload = await response.json() as MeetingResult | { detail?: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "会议处理失败");
      const job = payload as { task_id: string; report_path?: string };
      taskId = job.task_id;
      cancelledMeetingTask.current = null;
      setActiveMeetingTask(taskId);
      setStatus("会议处理中，请稍候...");
      const completed = await waitForMeeting(job.task_id, job.report_path ?? "", file.name);
      setMeeting(completed);
      localStorage.setItem("astra:lastMeetingTask", completed.task_id);
      setStatus("会议纪要生成完成");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "会议处理失败");
    } finally {
      setBusy(false);
      if (taskId) setActiveMeetingTask((current) => current === taskId ? null : current);
    }
  }

  async function cancelMeeting() {
    if (!activeMeetingTask) return;
    const taskId = activeMeetingTask;
    try {
      const response = await fetch(`/api/meeting/${encodeURIComponent(taskId)}`, { method: "DELETE" });
      const payload = await response.json() as { detail?: string; status?: string };
      if (!response.ok) throw new Error(payload.detail ?? "取消会议处理失败");
      cancelledMeetingTask.current = taskId;
      meetingSocket.current?.close();
      meetingSocket.current = null;
      setActiveMeetingTask(null);
      setBusy(false);
      setStatus(payload.status === "stopped" ? "会议已取消" : "会议任务已结束");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "取消会议处理失败");
    }
  }

  async function refreshSpeakers() {
    try {
      const response = await fetch("/api/speakers");
      const payload = await response.json() as { items?: SpeakerProfile[]; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "无法读取声纹档案");
      const items = payload.items ?? [];
      setSpeakers(items);
      setRenaming(Object.fromEntries(items.map((speaker) => [speaker.speaker_id, speaker.display_name])));
      await refreshSampleLists(items);
      setSpeakerStatus(items.length > 0 ? `已加载 ${items.length} 个声纹档案` : "暂无声纹档案");
    } catch (error) {
      setSpeakerStatus(error instanceof Error ? error.message : "声纹档案加载失败");
    }
  }

  async function refreshSampleLists(items: SpeakerProfile[]) {
    const entries = await Promise.all(items.map(async (speaker) => {
      const response = await fetch(`/api/speakers/${encodeURIComponent(speaker.speaker_id)}/samples`);
      if (!response.ok) return [speaker.speaker_id, []] as const;
      const payload = await response.json() as { items?: SpeakerSample[] };
      return [speaker.speaker_id, payload.items ?? []] as const;
    }));
    setSpeakerSamples(Object.fromEntries(entries));
  }

  function connectNotifications() {
    notificationSocket.current?.close();
    const connection = new WebSocket(notificationEventUrl());
    notificationSocket.current = connection;
    connection.onmessage = (message) => {
      try {
        const payload = JSON.parse(message.data) as NotificationEvent;
        if (payload.type === "notifications") setSpeakerNotifications(payload.items ?? []);
      } catch {
        setSpeakerNotifications([]);
      }
    };
    connection.onclose = () => {
      if (notificationSocket.current === connection) notificationSocket.current = null;
    };
  }

  async function createSpeaker(name: string) {
    const displayName = name.trim();
    if (!displayName) {
      setSpeakerStatus("请输入说话人名称");
      return;
    }
    try {
      const response = await fetch("/api/speakers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: displayName }),
      });
      const payload = await response.json() as SpeakerProfile | { detail?: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "创建失败");
      setSpeakerName("");
      setSpeakerStatus(`已创建 ${displayName}`);
      await refreshSpeakers();
    } catch (error) {
      setSpeakerStatus(error instanceof Error ? error.message : "创建声纹档案失败");
    }
  }

  async function createDefaultSpeakers() {
    for (let index = 1; index <= 8; index += 1) {
      const name = `S${index}`;
      if (!speakers.some((speaker) => speaker.display_name === name)) {
        await createSpeaker(name);
      }
    }
    await refreshSpeakers();
  }

  async function renameSpeaker(speaker: SpeakerProfile) {
    const nextName = (renaming[speaker.speaker_id] ?? "").trim();
    if (!nextName || nextName === speaker.display_name) return;
    try {
      const response = await fetch(`/api/speakers/${encodeURIComponent(speaker.speaker_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: nextName }),
      });
      const payload = await response.json() as SpeakerProfile | { detail?: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "改名失败");
      setSpeakerStatus(`已改名为 ${nextName}`);
      await refreshSpeakers();
    } catch (error) {
      setSpeakerStatus(error instanceof Error ? error.message : "声纹改名失败");
    }
  }

  async function approveSpeaker(speaker: SpeakerProfile) {
    try {
      const response = await fetch(`/api/speakers/${encodeURIComponent(speaker.speaker_id)}/approve`, {
        method: "POST",
      });
      const payload = await response.json() as SpeakerProfile | { detail?: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "审核失败");
      setSpeakerStatus(`已审核 ${speaker.display_name}，后续会议将参与匹配`);
      await refreshSpeakers();
    } catch (error) {
      setSpeakerStatus(error instanceof Error ? error.message : "声纹审核失败");
    }
  }

  async function disableSpeaker(speaker: SpeakerProfile) {
    try {
      const response = await fetch(`/api/speakers/${encodeURIComponent(speaker.speaker_id)}`, {
        method: "DELETE",
      });
      const payload = await response.json() as SpeakerProfile | { detail?: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "禁用失败");
      setSpeakerStatus(`已禁用 ${speaker.display_name}`);
      await refreshSpeakers();
    } catch (error) {
      setSpeakerStatus(error instanceof Error ? error.message : "禁用声纹失败");
    }
  }

  async function uploadSpeakerSample(speaker: SpeakerProfile, files: FileList | null) {
    const sample = files?.[0];
    if (!sample) return;
    const form = new FormData();
    form.append("file", sample);
    setSpeakerStatus(`正在注册 ${speaker.display_name} 的样本...`);
    try {
      const response = await fetch(`/api/speakers/${encodeURIComponent(speaker.speaker_id)}/samples`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json() as { detail?: string; speech_duration_s?: number; quality_score?: number };
      if (!response.ok) throw new Error(payload.detail ?? "样本注册失败");
      setSpeakerStatus(
        `已注册 ${speaker.display_name}: 语音 ${payload.speech_duration_s?.toFixed(1) ?? "-"}s, 质量 ${payload.quality_score?.toFixed(2) ?? "-"}`,
      );
      await refreshSpeakers();
    } catch (error) {
      setSpeakerStatus(error instanceof Error ? error.message : "样本注册失败");
    }
  }

  const currentTemplate = promptTemplates.find((template) => template.id === promptTemplate);

  return (
    <section className="view" aria-label="音频工作台">
      <div className="page-head">
        <div className="page-head__text">
          <div className="eyebrow">02 · 音频工作台</div>
          <h1>上传录音，得到可追溯的转写与纪要</h1>
          <p>上传本地音频，按需执行基础转写、流式修正或完整会议纪要。所有处理都在本机完成。</p>
        </div>
        <div className="page-head__actions">
          <span className={`badge ${speakerNotifications.length > 0 ? "badge--warning" : "badge--neutral"}`}>
            {speakerNotifications.length > 0 && <span className="badge__dot" />}
            {speakerNotifications.length > 0 ? `待审核 ${speakerNotifications.length}` : "无新提醒"}
          </span>
        </div>
      </div>

      <div className="work-grid">
        <div className="col">
          {/* 上传 + 配置 */}
          <form className="card" onSubmit={upload}>
            <div className="card__head"><h3>新建任务</h3></div>

            <label className="dropzone" role="button" tabIndex={0}>
              <div className="dropzone__icon">⇪</div>
              <div className="dropzone__title">{file ? file.name : "拖入音频文件，或点击选择"}</div>
              <div className="dropzone__hint">
                {file
                  ? `${file.size.toLocaleString()} bytes · 已就绪`
                  : "支持 m4a / wav / mp3 / flac / aac / mov / mp4"}
              </div>
              <input type="file" accept="audio/*,.wav" onChange={selectFile} />
            </label>

            {file && (
              <div className="filecard" style={{ marginTop: "var(--space-4)" }}>
                <div className="filecard__icon">♪</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="filecard__name">{file.name}</div>
                  <div className="filecard__meta">{file.size.toLocaleString()} bytes · 已就绪</div>
                </div>
                <button
                  type="button"
                  className="btn btn--ghost btn--sm"
                  onClick={() => {
                    setFile(null);
                    setStatus("选择一个音频文件开始测试");
                  }}
                >
                  移除
                </button>
              </div>
            )}

            <div className="divider" />

            <div className="form-grid">
              <div className="field">
                <label className="field__label" htmlFor="w-topic">会议主题</label>
                <input
                  className="input"
                  id="w-topic"
                  type="text"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="留空时由模型根据内容推断"
                />
              </div>
              <div className="field">
                <label className="field__label" htmlFor="w-tpl">提示词模板</label>
                <select
                  className="select"
                  id="w-tpl"
                  value={promptTemplate}
                  onChange={(e) => setPromptTemplate(e.target.value)}
                >
                  {(promptTemplates.length > 0
                    ? promptTemplates
                    : [{ id: "standard", name: "标准纪要", description: "", chunk_system_prompt: "", merge_system_prompt: "" }]
                  ).map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.name}
                    </option>
                  ))}
                </select>
                <span className="field__hint">
                  {currentTemplate?.description ?? "提取会议结论、行动项和待确认事项"}
                </span>
                <div className="page-head__actions" style={{ marginTop: "var(--space-1)" }}>
                  <button type="button" className="btn btn--ghost btn--sm" onClick={openNewPromptTemplate}>新建自定义</button>
                  {currentTemplate?.editable && (
                    <>
                      <button type="button" className="btn btn--ghost btn--sm" onClick={openEditPromptTemplate}>编辑</button>
                      <button type="button" className="btn btn--ghost btn--sm" onClick={deletePromptTemplate}>删除</button>
                    </>
                  )}
                </div>
              </div>
            </div>

            <h4 style={{ margin: "var(--space-5) 0 var(--space-3)" }}>处理方式</h4>
            <div className="modes">
              <button
                type="button"
                className="mode"
                aria-pressed={!separateSpeakers && !streamSeparate}
                onClick={() => {
                  setSeparateSpeakers(false);
                  setStreamSeparate(false);
                }}
              >
                <span className="mode__top">
                  <span className="mode__name">标准处理</span>
                  <span className="mode__check">✓</span>
                </span>
                <span className="mode__desc">默认的转写与纪要流程，不启用重叠语音分离。</span>
              </button>
              <button
                type="button"
                className="mode"
                aria-pressed={streamSeparate}
                onClick={() => setStreamSeparate((v) => !v)}
              >
                <span className="mode__top">
                  <span className="mode__name">流式分离</span>
                  <span className="mode__check">✓</span>
                </span>
                <span className="mode__desc">流式修正时启用 FLASepformer 双说话人分离（8 kHz）。</span>
              </button>
              <button
                type="button"
                className="mode"
                aria-pressed={separateSpeakers}
                onClick={() => setSeparateSpeakers((v) => !v)}
              >
                <span className="mode__top">
                  <span className="mode__name">会议分离</span>
                  <span className="mode__check">✓</span>
                </span>
                <span className="mode__desc">生成会议纪要时启用重叠语音分离；未配置模型则保持原流程。</span>
              </button>
            </div>

            <div className="divider" />

            <div className="page-head__actions">
              <button type="submit" className="btn btn--secondary" disabled={busy || !file}>
                {busy ? "转写中…" : "基础转写"}
              </button>
              <button type="button" className="btn btn--secondary" disabled={busy || !file} onClick={uploadWithCorrection}>
                流式修正
              </button>
              <button type="button" className="btn btn--primary btn--lg" disabled={busy || !file} onClick={runMeeting}>
                {busy ? "处理中…" : "生成会议纪要"}
              </button>
              {activeMeetingTask && (
                <button type="button" className="btn btn--danger" onClick={cancelMeeting}>
                  取消会议处理
                </button>
              )}
            </div>
            <p className="card__hint" style={{ marginTop: "var(--space-3)" }}>{status}</p>
          </form>

          {/* 自定义模板编辑器 */}
          {promptEditor && (
            <div className="card" aria-live="polite">
              <div className="card__head">
                <h3>{promptEditor.id ? "编辑自定义模板" : "新建自定义模板"}</h3>
                <button type="button" className="btn btn--ghost btn--sm" onClick={() => setPromptEditor(null)}>取消</button>
              </div>
              <div className="form-grid">
                <div className="field">
                  <label className="field__label">模板名称</label>
                  <input className="input" value={promptEditor.name} maxLength={80} onChange={(event) => updatePromptEditor("name", event.target.value)} placeholder="例如：产品评审纪要" />
                </div>
                <div className="field">
                  <label className="field__label">模板说明</label>
                  <input className="input" value={promptEditor.description} maxLength={200} onChange={(event) => updatePromptEditor("description", event.target.value)} placeholder="说明这个模板适用的会议类型" />
                </div>
              </div>
              <div className="field" style={{ marginTop: "var(--space-4)" }}>
                <label className="field__label">分段提取系统提示词</label>
                <textarea className="textarea" value={promptEditor.chunk_system_prompt} maxLength={20000} onChange={(event) => updatePromptEditor("chunk_system_prompt", event.target.value)} rows={6} />
              </div>
              <div className="field" style={{ marginTop: "var(--space-4)" }}>
                <label className="field__label">最终合并系统提示词</label>
                <textarea className="textarea" value={promptEditor.merge_system_prompt} maxLength={20000} onChange={(event) => updatePromptEditor("merge_system_prompt", event.target.value)} rows={6} />
              </div>
              <div className="page-head__actions" style={{ marginTop: "var(--space-4)" }}>
                <button
                  type="button"
                  className="btn btn--primary btn--sm"
                  disabled={!promptEditor.name.trim() || !promptEditor.chunk_system_prompt.trim() || !promptEditor.merge_system_prompt.trim()}
                  onClick={savePromptTemplate}
                >
                  保存模板
                </button>
                <span className="card__hint">自定义模板保存在服务端，内置模板不可覆盖。</span>
              </div>
              {promptEditorStatus && <p className="field__error" style={{ marginTop: "var(--space-2)" }}>{promptEditorStatus}</p>}
            </div>
          )}

          {/* 结果 */}
          {meeting && (
            <div className="card" aria-live="polite">
              <div className="card__head">
                <span className="badge badge--success">已完成</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="taskcard__title">{meeting.filename}</div>
                  <div className="taskcard__id">
                    {meeting.task_id} · {Math.round(meeting.duration_s / 60)}分{Math.round(meeting.duration_s % 60)}秒 ·
                    说话人 {meeting.speakers.length > 0 ? meeting.speakers.join("/") : "未分离"} · {meeting.segments} 段
                  </div>
                </div>
              </div>
              {meeting.summary && <MarkdownContent source={meeting.summary} />}
              {meeting.timeline_preview && (
                <details>
                  <summary>逐字稿预览</summary>
                  <pre className="timeline-pre">{meeting.timeline_preview}</pre>
                </details>
              )}
              {meeting.separation_audio_urls && meeting.separation_audio_urls.length > 0 && (
                <div className="spk-samples" style={{ marginTop: "var(--space-4)" }}>
                  <strong className="card__hint">分离音轨复听</strong>
                  {meeting.separation_audio_urls.map((audioUrl, index) => (
                    <div key={audioUrl}>
                      <div className="filecard__meta">来源 {index + 1}</div>
                      <audio controls preload="none" src={audioUrl} />
                    </div>
                  ))}
                </div>
              )}
              {meeting.overlap_detection && <OverlapDetectionView detection={meeting.overlap_detection} />}
              <div className="divider" />
              <div className="kv">
                <span className="kv__k">完整报告</span><span className="kv__v">{meeting.report_path}</span>
                {meeting.training_artifacts && (
                  <>
                    <span className="kv__k">训练候选</span>
                    <span className="kv__v">{meeting.training_artifacts.candidate_segments ?? 0} 段 · {meeting.training_artifacts.training_candidates}</span>
                    <span className="kv__k">逐字稿审校</span><span className="kv__v">{meeting.training_artifacts.transcript_segments}</span>
                    <span className="kv__k">音频片段</span><span className="kv__v">{meeting.training_artifacts.clips_dir}</span>
                  </>
                )}
              </div>
              <div className="page-head__actions" style={{ marginTop: "var(--space-4)" }}>
                <a className="btn btn--secondary btn--sm" href={`/review?task_id=${encodeURIComponent(meeting.task_id)}`}>打开逐段审校</a>
                <a className="btn btn--ghost btn--sm" href="/training">打开训练设置</a>
              </div>
            </div>
          )}

          {result && (
            <div className="card" aria-live="polite">
              <div className="card__head"><h4>转写结果</h4></div>
              <p className="card__hint">{result.filename} · {result.bytes.toLocaleString()} bytes</p>
              <p className="spine-text" style={{ marginTop: "var(--space-3)" }}>{result.text || "未识别到文本"}</p>
            </div>
          )}

          {correction && (
            <div className="card" aria-live="polite">
              <div className="card__head"><h4>流式修正</h4></div>
              <p className="spine-text">{correction}</p>
            </div>
          )}

          {enhancementStatus.length > 0 && (
            <div className="card" aria-live="polite">
              <div className="card__head"><h4>音频增强</h4></div>
              {enhancementStatus.map((stage) => (
                <p className="card__hint" key={stage.stage_name}>
                  {stage.stage_name} · {stage.status} · {stage.latency_ms.toFixed(1)} ms
                  {stage.fallback_reason ? ` · ${stage.fallback_reason}` : ""}
                </p>
              ))}
            </div>
          )}

          {separationStatus && (
            <div className="card" aria-live="polite">
              <div className="card__head"><h4>流式分离</h4></div>
              <p className="card__hint">
                {separationStatus.stage_name} · {separationStatus.status} · {separationStatus.output_count} 路 · {separationStatus.latency_ms.toFixed(1)} ms
                {separationStatus.fallback_reason ? ` · ${separationStatus.fallback_reason}` : ""}
              </p>
            </div>
          )}

          {correctionTimeline.length > 0 && (
            <div className="card" aria-live="polite">
              <div className="card__head"><h4>修正时间线</h4></div>
              <div className="spine">
                {correctionTimeline.map((segment) => (
                  <div className="spine-row" key={segment.index}>
                    <div className="spine-time">{formatTime(segment.start)}</div>
                    <div className="spine-body">
                      {segment.source_index !== undefined && (
                        <div className="spine-speaker">来源 {segment.source_index + 1}</div>
                      )}
                      <div className="spine-text">{segment.text}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 右栏 */}
        <div className="col">
          <div className="card">
            <div className="card__head">
              <h4>提示词模板</h4>
              <button type="button" className="btn btn--ghost btn--sm" onClick={openNewPromptTemplate}>新建</button>
            </div>
            <div>
              {(promptTemplates.length > 0 ? promptTemplates : []).map((template) => (
                <div className="spk-row" key={template.id}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="spk-name">{template.name}</div>
                    <div className="spk-meta">{template.description || (template.builtin ? "内置模板" : "自定义模板")}</div>
                  </div>
                  {template.editable ? (
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm"
                      onClick={() => {
                        setPromptTemplate(template.id);
                        openEditPromptTemplate();
                      }}
                    >
                      编辑
                    </button>
                  ) : (
                    <span className="badge badge--neutral">内置</span>
                  )}
                </div>
              ))}
              {promptTemplates.length === 0 && <p className="card__hint">正在加载模板…</p>}
            </div>
          </div>

          <div className="card" aria-live="polite">
            <div className="card__head">
              <h4>说话人档案</h4>
              <button type="button" className="btn btn--ghost btn--sm" onClick={refreshSpeakers}>刷新</button>
              <button type="button" className="btn btn--ghost btn--sm" onClick={createDefaultSpeakers}>创建 S1-S8</button>
            </div>
            <div className="setting-inline" style={{ marginBottom: "var(--space-3)" }}>
              <input
                className="input"
                type="text"
                value={speakerName}
                onChange={(event) => setSpeakerName(event.target.value)}
                placeholder="新说话人名称"
              />
              <button type="button" className="btn btn--secondary btn--sm" onClick={() => createSpeaker(speakerName)}>创建</button>
            </div>
            <p className="card__hint">{speakerStatus}</p>
            {speakerNotifications.length > 0 && (
              <div className="alert alert--warning" role="status" style={{ marginTop: "var(--space-3)" }}>
                <span>◆</span>
                <div className="alert__body">
                  {speakerNotifications.map((notification) => (
                    <div className="alert__desc" key={notification.id}>{notification.message}</div>
                  ))}
                </div>
              </div>
            )}
            <div style={{ marginTop: "var(--space-2)" }}>
              {speakers.map((speaker) => {
                const pending = speaker.status === "pending_review";
                return (
                  <div className="spk-row" key={speaker.speaker_id} style={{ flexWrap: "wrap" }}>
                    <div className={`spk-avatar${pending ? " spk-avatar--pending" : ""}`}>
                      {speaker.display_name.slice(0, 2)}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div className="spk-name">
                        {speaker.display_name}
                        {pending && (
                          <span className="badge badge--warning" style={{ marginLeft: "var(--space-2)" }}>待审核</span>
                        )}
                      </div>
                      <div className="spk-meta">{speaker.sample_count} 个样本 · {speaker.embedding_model}</div>
                      <div className="spk-samples">
                        {(speakerSamples[speaker.speaker_id] ?? []).map((sample) => (
                          <div key={sample.sample_id}>
                            {sample.audio_url ? (
                              <audio controls preload="none" src={sample.audio_url} />
                            ) : (
                              <span className="spk-meta">历史样本无音频</span>
                            )}
                            <div className="spk-meta">{sample.original_filename} · {sample.duration_s.toFixed(1)}s</div>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="spk-actions" style={{ width: "100%" }}>
                      <input
                        className="input input--mono"
                        type="text"
                        value={renaming[speaker.speaker_id] ?? speaker.display_name}
                        onChange={(event) => setRenaming((current) => ({ ...current, [speaker.speaker_id]: event.target.value }))}
                        aria-label={`${speaker.display_name} 的新名称`}
                      />
                      <button type="button" className="btn btn--secondary btn--sm" onClick={() => renameSpeaker(speaker)}>改名</button>
                      {pending && (
                        <button type="button" className="btn btn--primary btn--sm" onClick={() => approveSpeaker(speaker)}>审核通过</button>
                      )}
                      <label className="sample-upload">
                        <span className="btn btn--ghost btn--sm">样本</span>
                        <input type="file" accept="audio/*,.wav" onChange={(event) => uploadSpeakerSample(speaker, event.target.files)} />
                      </label>
                      <button type="button" className="btn btn--danger btn--sm" onClick={() => disableSpeaker(speaker)}>禁用</button>
                    </div>
                  </div>
                );
              })}
              {speakers.length === 0 && <p className="card__hint">暂无声纹档案。</p>}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
