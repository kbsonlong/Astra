import { ChangeEvent, FormEvent, Fragment, ReactNode, useEffect, useRef, useState } from "react";

type TranscriptionResult = {
  filename: string;
  bytes: number;
  text: string;
};


type TrainingConfig = {
  model_path: string;
  train_file: string;
  eval_file: string;
  output_dir: string;
  device: "auto" | "cuda" | "mps" | "cpu";
  precision: "bf16" | "fp16" | "fp32";
  batch_size: number;
  grad_acc: number;
  learning_rate: number;
  epochs: number;
  save_steps: number;
  save_total_limit: number;
  num_workers: number;
  pin_memory: boolean;
  persistent_workers: boolean;
  prefetch_factor: number;
  resume_from: string;
  resume_latest: boolean;
};
type TrainingStatus = {
  status: "idle" | "processing" | "done" | "failed";
  task_id?: string;
  returncode?: number | null;
  log_path?: string;
  output_dir?: string;
};
type TimelineSegment = {
  index: number;
  start: number;
  end: number;
  text: string;
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
  status: "processing" | "done" | "failed";
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
  const [meeting, setMeeting] = useState<MeetingResult | null>(null);
  const [speakers, setSpeakers] = useState<SpeakerProfile[]>([]);
  const [speakerName, setSpeakerName] = useState("");
  const [speakerStatus, setSpeakerStatus] = useState("声纹档案未加载");
  const [renaming, setRenaming] = useState<Record<string, string>>({});
  const [speakerNotifications, setSpeakerNotifications] = useState<SpeakerNotification[]>([]);
  const [speakerSamples, setSpeakerSamples] = useState<Record<string, SpeakerSample[]>>({});
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig | null>(null);
  const [trainingStatus, setTrainingStatus] = useState("训练参数未加载");
  const [trainingBusy, setTrainingBusy] = useState(false);
  const [trainingStatusInfo, setTrainingStatusInfo] = useState<TrainingStatus>({ status: "idle" });
  const meetingSocket = useRef<WebSocket | null>(null);
  const notificationSocket = useRef<WebSocket | null>(null);

  useEffect(() => {
    void refreshSpeakers();
    void refreshTrainingConfig();
    connectNotifications();
    return () => {
      meetingSocket.current?.close();
      notificationSocket.current?.close();
    };
  }, []);

  useEffect(() => {
    if (trainingStatusInfo.status !== "processing") return undefined;
    const timer = window.setInterval(() => { void refreshTrainingStatus(); }, 2000);
    return () => window.clearInterval(timer);
  }, [trainingStatusInfo.status]);

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setResult(null);
    setCorrection("");
    setCorrectionTimeline([]);
    setMeeting(null);
    meetingSocket.current?.close();
    meetingSocket.current = null;
    setStatus(event.target.files?.[0]?.name ?? "选择一个音频文件开始测试");
  }

  async function refreshTrainingConfig() {
    try {
      const response = await fetch("/api/training/config");
      const payload = await response.json() as { config?: TrainingConfig; detail?: string };
      if (!response.ok || !payload.config) throw new Error(payload.detail ?? "无法读取训练参数");
      setTrainingConfig(payload.config);
      setTrainingStatus("已加载离线训练参数");
    } catch (error) {
      setTrainingStatus(error instanceof Error ? error.message : "训练参数加载失败");
    }
  }

  async function saveTrainingConfig() {
    if (!trainingConfig) return;
    setTrainingBusy(true);
    setTrainingStatus("正在保存训练参数...");
    try {
      const response = await fetch("/api/training/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trainingConfig),
      });
      const payload = await response.json() as { config?: TrainingConfig; detail?: string };
      if (!response.ok || !payload.config) throw new Error(payload.detail ?? "训练参数保存失败");
      setTrainingConfig(payload.config);
      setTrainingStatus("训练参数已保存；不会自动启动训练或热切换模型");
    } catch (error) {
      setTrainingStatus(error instanceof Error ? error.message : "训练参数保存失败");
    } finally {
      setTrainingBusy(false);
    }
  }

  async function refreshTrainingStatus() {
    try {
      const response = await fetch("/api/training/status");
      const payload = await response.json() as TrainingStatus & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "训练状态读取失败");
      setTrainingStatusInfo(payload);
      if (payload.status === "processing") setTrainingStatus(`训练进行中 · ${payload.task_id ?? ""}`);
      if (payload.status === "done") setTrainingStatus("训练完成，适配器已写入输出目录");
      if (payload.status === "failed") setTrainingStatus(`训练失败（退出码 ${payload.returncode ?? "未知"}），请查看日志`);
      if (payload.status === "idle") setTrainingStatus("当前没有运行中的训练任务");
    } catch (error) {
      setTrainingStatus(error instanceof Error ? error.message : "训练状态读取失败");
    }
  }

  async function startTraining() {
    if (!trainingConfig) return;
    setTrainingBusy(true);
    setTrainingStatus("正在启动后台训练...");
    try {
      const response = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trainingConfig),
      });
      const payload = await response.json() as TrainingStatus & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "训练启动失败");
      setTrainingStatusInfo(payload);
      setTrainingStatus(`训练已启动 · ${payload.task_id ?? ""}`);
    } catch (error) {
      setTrainingStatus(error instanceof Error ? error.message : "训练启动失败");
    } finally {
      setTrainingBusy(false);
    }
  }

  async function stopTraining() {
    setTrainingBusy(true);
    try {
      const response = await fetch("/api/training/stop", { method: "POST" });
      const payload = await response.json() as TrainingStatus & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "训练停止失败");
      setTrainingStatusInfo(payload);
      setTrainingStatus("已请求停止训练");
    } catch (error) {
      setTrainingStatus(error instanceof Error ? error.message : "训练停止失败");
    } finally {
      setTrainingBusy(false);
    }
  }

  function updateTrainingConfig<K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) {
    setTrainingConfig((current) => current ? { ...current, [key]: value } : current);
  }

  function numberValue(value: string, fallback: number): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function renderTrainingNumber(
    key: keyof TrainingConfig,
    label: string,
    step = "1",
  ) {
    if (!trainingConfig) return null;
    const value = trainingConfig[key];
    return (
      <label className="training-field" key={String(key)}>
        <span>{label}</span>
        <input
          type="number"
          step={step}
          value={String(value)}
          onChange={(event) => updateTrainingConfig(key, numberValue(event.target.value, Number(value)) as TrainingConfig[typeof key])}
        />
      </label>
    );
  }

  function renderTrainingText(key: "model_path" | "train_file" | "eval_file" | "output_dir" | "resume_from", label: string) {
    if (!trainingConfig) return null;
    return (
      <label className="training-field training-field-wide" key={key}>
        <span>{label}</span>
        <input
          type="text"
          value={trainingConfig[key]}
          onChange={(event) => updateTrainingConfig(key, event.target.value)}
        />
      </label>
    );
  }

  function renderTrainingCheckbox(key: "pin_memory" | "persistent_workers" | "resume_latest", label: string) {
    if (!trainingConfig) return null;
    return (
      <label className="training-check" key={key}>
        <input
          type="checkbox"
          checked={trainingConfig[key]}
          onChange={(event) => updateTrainingConfig(key, event.target.checked)}
        />
        <span>{label}</span>
      </label>
    );
  }

  function renderTrainingPanel() {
    return (
      <section className="training-panel" aria-live="polite">
        <div className="speaker-panel-head">
          <div>
            <span className="eyebrow">QWEN3-ASR · OFFLINE SFT</span>
            <h2>领域微调参数</h2>
          </div>
          <div className="speaker-actions">
            <button type="button" className="secondary-action compact-button" onClick={refreshTrainingConfig} disabled={trainingBusy}>重载</button>
            <button type="button" className="secondary-action compact-button" onClick={refreshTrainingStatus} disabled={trainingBusy}>查状态</button>
            <button type="button" className="compact-button" onClick={saveTrainingConfig} disabled={!trainingConfig || trainingBusy}>保存参数</button>
            {trainingStatusInfo.status === "processing" ? (
              <button type="button" className="danger-action compact-button" onClick={stopTraining} disabled={trainingBusy}>停止训练</button>
            ) : (
              <button type="button" className="compact-button" onClick={startTraining} disabled={!trainingConfig || trainingBusy}>启动训练</button>
            )}
          </div>
        </div>
        <p className="training-note">训练在独立后台进程运行，不阻塞 API；需要使用已安装 mlx-tune 的 Python。默认训练基座是非量化 Qwen3-ASR-0.6B，会议推理仍使用独立的 MLX 4bit 模型。</p>
        <p className="speaker-status">{trainingStatus}</p>
        {trainingConfig && (
          <div className="training-grid">
            {renderTrainingText("model_path", "训练基座模型")}
            {renderTrainingText("train_file", "训练 JSONL")}
            {renderTrainingText("eval_file", "验证 JSONL")}
            {renderTrainingText("output_dir", "输出目录")}
            <label className="training-field"><span>设备</span><select value={trainingConfig.device} onChange={(event) => updateTrainingConfig("device", event.target.value as TrainingConfig["device"])}><option value="cuda">CUDA</option><option value="mps">Apple MPS（实验）</option><option value="cpu">CPU（不推荐）</option><option value="auto">自动</option></select></label>
            <label className="training-field"><span>精度</span><select value={trainingConfig.precision} onChange={(event) => updateTrainingConfig("precision", event.target.value as TrainingConfig["precision"])}><option value="bf16">BF16</option><option value="fp16">FP16</option><option value="fp32">FP32</option></select></label>
            {renderTrainingNumber("batch_size", "Batch size")}
            {renderTrainingNumber("grad_acc", "梯度累积")}
            {renderTrainingNumber("learning_rate", "学习率", "0.000001")}
            {renderTrainingNumber("epochs", "Epochs")}
            {renderTrainingNumber("save_steps", "保存步数")}
            {renderTrainingNumber("save_total_limit", "保留 checkpoint")}
            {renderTrainingNumber("num_workers", "数据线程")}
            {renderTrainingNumber("prefetch_factor", "预取因子")}
            {renderTrainingText("resume_from", "恢复 checkpoint（可选）")}
            <div className="training-checks">
              {renderTrainingCheckbox("pin_memory", "启用 pin memory")}
              {renderTrainingCheckbox("persistent_workers", "保持数据线程")}
              {renderTrainingCheckbox("resume_latest", "自动恢复最新 checkpoint")}
            </div>
          </div>
        )}
      </section>
    );
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
        if (current.status === "failed") {
          finished = true;
          connection.close();
          reject(new Error(current.error ?? "会议处理失败，请查看服务端日志"));
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
        if (!finished) reject(new Error("会议状态 WebSocket 已断开"));
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
    try {
      const response = await fetch("/api/meeting/process", { method: "POST", body: form });
      const payload = await response.json() as MeetingResult | { detail?: string };
      if (!response.ok) throw new Error("detail" in payload ? payload.detail : "会议处理失败");
      const job = payload as { task_id: string; report_path?: string };
      setStatus("会议处理中，请稍候...");
      const completed = await waitForMeeting(job.task_id, job.report_path ?? "", file.name);
      setMeeting(completed);
      setStatus("会议纪要生成完成");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "会议处理失败");
    } finally {
      setBusy(false);
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
          <a className="nav-link active" href="/upload">音频工作台</a>
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
        </section>

        <section className="action-panel">
          <button type="submit" disabled={busy || !file}>{busy ? "转写中..." : "基础转写"}</button>
          <button type="button" className="secondary-action" disabled={busy || !file} onClick={uploadWithCorrection}>流式修正</button>
          <button type="button" className="meeting-btn" disabled={busy || !file} onClick={runMeeting}>
            {busy ? "处理中..." : "生成会议纪要"}
          </button>
        </section>
      </form>

      {renderTrainingPanel()}

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
