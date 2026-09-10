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
  const [status, setStatus] = useState("选择一个音频文件开始测试");
  const [busy, setBusy] = useState(false);
  const [topic, setTopic] = useState("");
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
    setStatus("正在转写并流式修正...");
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/api/transcribe/stream", { method: "POST", body: form });
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
            index?: number; start?: number; end?: number;
          };
          if (payload.type === "asr_final") setResult({ filename: file.name, bytes: file.size, text: payload.text ?? "" });
          if (payload.type === "asr_segment" && payload.index !== undefined) {
            const segment = { index: payload.index, start: payload.start ?? 0, end: payload.end ?? 0, text: payload.text ?? "" };
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

  return (
    <main className="app-shell upload-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">A</span>
          <span>
            <strong>Astra</strong>
            <small>Audio Workbench</small>
          </span>
        </a>
        <nav className="nav-actions" aria-label="Astra tools">
          <a className="nav-link" href="/">实时通话</a>
          <a className="nav-link active" href="/upload">会议工作台</a>
          <a className="nav-link" href="/review">逐段审校</a>
          <a className="nav-link" href="/training">训练设置</a>
          <a className="nav-link" href="/settings">管理设置</a>
          <span className={`notification-indicator${speakerNotifications.length > 0 ? " has-notifications" : ""}`} title="声纹审核提醒">
            {speakerNotifications.length > 0 ? `待审核 ${speakerNotifications.length}` : "无新提醒"}
          </span>
        </nav>
      </header>

      <section className="workbench-hero">
        <div>
          <span className="eyebrow">ASR · CORRECTION · MEETING NOTES</span>
          <h1>把录音变成可审阅的文本资产</h1>
          <p>上传本地音频，按需执行转写、流式修正或会议纪要生成。</p>
        </div>
        <div className="status-board">
          <span>当前任务</span>
          <strong>{busy ? "处理中" : file ? "已选择文件" : "等待音频"}</strong>
          <small>{status}</small>
        </div>
      </section>

      <form className="workbench-grid" onSubmit={upload}>
        <section className="upload-panel">
          <label className="file-drop">
            <span className="file-drop-icon">+</span>
            <strong>{file ? file.name : "选择音频文件"}</strong>
            <small>{file ? `${file.size.toLocaleString()} bytes` : "支持浏览器可读取的音频格式"}</small>
            <input type="file" accept="audio/*,.wav" onChange={selectFile} />
          </label>

          <label className="field">
            <span>会议主题</span>
            <input type="text" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="例如：云资源成本优化周会" />
          </label>

          <label className="field">
            <span>纪要提示词模板</span>
            <select value={promptTemplate} onChange={(e) => setPromptTemplate(e.target.value)}>
              {(promptTemplates.length > 0 ? promptTemplates : [{
                id: "standard",
                name: "标准纪要",
                description: "",
                chunk_system_prompt: "",
                merge_system_prompt: "",
              }]).map((template) => (
                <option key={template.id} value={template.id}>{template.name}</option>
              ))}
            </select>
            <small className="field-hint">
              {promptTemplates.find((template) => template.id === promptTemplate)?.description ?? "提取会议结论、行动项和待确认事项"}
            </small>
            <span className="prompt-template-actions">
              <button type="button" className="secondary-action compact-button" onClick={openNewPromptTemplate}>新建自定义</button>
              {promptTemplates.find((template) => template.id === promptTemplate)?.editable && (
                <>
                  <button type="button" className="secondary-action compact-button" onClick={openEditPromptTemplate}>编辑</button>
                  <button type="button" className="danger-action compact-button" onClick={deletePromptTemplate}>删除</button>
                </>
              )}
            </span>
          </label>

          <div className="field field-checkbox">
            <span>多人重叠语音</span>
            <label>
              <input
                type="checkbox"
                checked={separateSpeakers}
                onChange={(event) => setSeparateSpeakers(event.target.checked)}
              />
              <span>启用 FLASepformer 双说话人分离（离线、8 kHz、约 30 秒窗口）</span>
            </label>
            <small className="field-hint">
              仅会议纪要按钮生效；模型未配置时会保持原始混合音频流程。
            </small>
          </div>
        </section>

        <section className="action-panel">
          <button type="submit" disabled={busy || !file}>{busy ? "转写中..." : "基础转写"}</button>
          <button type="button" className="secondary-action" disabled={busy || !file} onClick={uploadWithCorrection}>流式修正</button>
          <button type="button" className="meeting-btn" disabled={busy || !file} onClick={runMeeting}>
            {busy ? "处理中..." : "生成会议纪要"}
          </button>
          {activeMeetingTask && (
            <button type="button" className="danger-action" onClick={cancelMeeting}>
              取消会议处理
            </button>
          )}
        </section>
      </form>

      {promptEditor && (
        <section className="prompt-editor-panel" aria-live="polite">
          <div className="prompt-editor-head">
            <div>
              <span className="eyebrow">CUSTOM MEETING PROMPT</span>
              <h2>{promptEditor.id ? "编辑自定义模板" : "新建自定义模板"}</h2>
            </div>
            <button type="button" className="secondary-action compact-button" onClick={() => setPromptEditor(null)}>取消</button>
          </div>
          <div className="prompt-editor-grid">
            <label className="field">
              <span>模板名称</span>
              <input value={promptEditor.name} maxLength={80} onChange={(event) => updatePromptEditor("name", event.target.value)} placeholder="例如：产品评审纪要" />
            </label>
            <label className="field">
              <span>模板说明</span>
              <input value={promptEditor.description} maxLength={200} onChange={(event) => updatePromptEditor("description", event.target.value)} placeholder="说明这个模板适用的会议类型" />
            </label>
            <label className="field prompt-textarea-field">
              <span>分段提取系统提示词</span>
              <textarea value={promptEditor.chunk_system_prompt} maxLength={20000} onChange={(event) => updatePromptEditor("chunk_system_prompt", event.target.value)} rows={8} />
            </label>
            <label className="field prompt-textarea-field">
              <span>最终合并系统提示词</span>
              <textarea value={promptEditor.merge_system_prompt} maxLength={20000} onChange={(event) => updatePromptEditor("merge_system_prompt", event.target.value)} rows={8} />
            </label>
          </div>
          <div className="prompt-editor-footer">
            <button type="button" disabled={!promptEditor.name.trim() || !promptEditor.chunk_system_prompt.trim() || !promptEditor.merge_system_prompt.trim()} onClick={savePromptTemplate}>保存模板</button>
            <small className="field-hint">自定义模板保存在服务端配置文件中，内置模板不可覆盖。</small>
            {promptEditorStatus && <small className="prompt-editor-status">{promptEditorStatus}</small>}
          </div>
        </section>
      )}

      <section className="speaker-panel" aria-live="polite">
        <div className="speaker-panel-head">
          <div>
            <span className="eyebrow">VOICEPRINT REGISTRY</span>
            <h2>声纹档案</h2>
          </div>
          <div className="speaker-actions">
            <button type="button" className="secondary-action compact-button" onClick={refreshSpeakers}>刷新</button>
            <button type="button" className="secondary-action compact-button" onClick={createDefaultSpeakers}>创建 S1-S8</button>
          </div>
        </div>
        <div className="speaker-create-row">
          <input
            type="text"
            value={speakerName}
            onChange={(event) => setSpeakerName(event.target.value)}
            placeholder="新说话人名称"
          />
          <button type="button" className="compact-button" onClick={() => createSpeaker(speakerName)}>创建</button>
        </div>
        <p className="speaker-status">{speakerStatus}</p>
        {speakerNotifications.length > 0 && (
          <div className="notification-list" role="status">
            {speakerNotifications.map((notification) => (
              <p key={notification.id}>{notification.message}</p>
            ))}
          </div>
        )}
        <div className="speaker-list">
          {speakers.map((speaker) => (
            <article className="speaker-row" key={speaker.speaker_id}>
              <div className="speaker-id-block">
                <strong>{speaker.display_name} {speaker.status === "pending_review" && <em>待审核</em>}</strong>
                <small>{speaker.sample_count} 个样本 · {speaker.embedding_model}</small>
                <div className="sample-list">
                  {(speakerSamples[speaker.speaker_id] ?? []).map((sample) => (
                    <div className="sample-item" key={sample.sample_id}>
                      {sample.audio_url ? (
                        <audio controls preload="none" src={sample.audio_url} />
                      ) : (
                        <span className="sample-unavailable">历史样本无音频</span>
                      )}
                      <small>{sample.original_filename} · {sample.duration_s.toFixed(1)}s</small>
                    </div>
                  ))}
                </div>
              </div>
              <input
                type="text"
                value={renaming[speaker.speaker_id] ?? speaker.display_name}
                onChange={(event) => setRenaming((current) => ({
                  ...current,
                  [speaker.speaker_id]: event.target.value,
                }))}
                aria-label={`${speaker.display_name} 的新名称`}
              />
              <button type="button" className="secondary-action compact-button" onClick={() => renameSpeaker(speaker)}>改名</button>
              {speaker.status === "pending_review" && (
                <button type="button" className="compact-button" onClick={() => approveSpeaker(speaker)}>审核通过</button>
              )}
              <label className="sample-upload">
                样本
                <input type="file" accept="audio/*,.wav" onChange={(event) => uploadSpeakerSample(speaker, event.target.files)} />
              </label>
              <button type="button" className="danger-action compact-button" onClick={() => disableSpeaker(speaker)}>禁用</button>
            </article>
          ))}
        </div>
      </section>

      <section className="results-stack">
        {meeting && (
          <article className="result-card meeting-result" aria-live="polite">
            <span className="eyebrow">MEETING NOTES · {meeting.task_id}</span>
            <p className="result-meta">
              {meeting.filename} · {Math.round(meeting.duration_s / 60)}分{Math.round(meeting.duration_s % 60)}秒 ·
              说话人 {meeting.speakers.length > 0 ? meeting.speakers.join("/") : "未分离"} · {meeting.segments} 段
            </p>
            {meeting.summary && <MarkdownContent source={meeting.summary} />}
            {meeting.timeline_preview && (
              <details open={false}>
                <summary>逐字稿预览</summary>
                <pre className="timeline-pre">{meeting.timeline_preview}</pre>
              </details>
            )}
            <p className="result-meta">完整报告: {meeting.report_path}</p>
            <p className="result-meta"><a className="review-link" href={`/review?task_id=${encodeURIComponent(meeting.task_id)}`}>打开逐段审校</a> · <a className="review-link" href="/training">打开训练设置</a></p>
            {meeting.training_artifacts && (
              <div className="result-meta">
                <strong>ASR 训练候选数据:</strong>{" "}
                {meeting.training_artifacts.candidate_segments ?? 0} 段，JSONL: {meeting.training_artifacts.training_candidates}
                <br />逐字稿审校: {meeting.training_artifacts.transcript_segments}
                <br />音频片段: {meeting.training_artifacts.clips_dir}
              </div>
            )}
          </article>
        )}
        {result && (
          <article className="result-card" aria-live="polite">
            <span className="eyebrow">TRANSCRIPTION RESULT</span>
            <p className="result-meta">{result.filename} · {result.bytes.toLocaleString()} bytes</p>
            <p className="result-text">{result.text || "未识别到文本"}</p>
          </article>
        )}
        {correction && (
          <article className="result-card correction-result" aria-live="polite">
            <span className="eyebrow">LLM CORRECTION STREAM</span>
            <p className="result-text">{correction}</p>
          </article>
        )}
        {correctionTimeline.length > 0 && (
          <article className="result-card correction-timeline" aria-live="polite">
            <span className="eyebrow">CORRECTION TIMELINE</span>
            {correctionTimeline.map((segment) => (
              <p className="timeline-line" key={segment.index}>
                <span>{formatTime(segment.start)} - {formatTime(segment.end)}</span>
                {segment.text}
              </p>
            ))}
          </article>
        )}
      </section>
    </main>
  );
}
