import { ChangeEvent, FormEvent, Fragment, ReactNode, useState } from "react";

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
};

type MeetingStatus = {
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

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setResult(null);
    setCorrection("");
    setCorrectionTimeline([]);
    setMeeting(null);
    setStatus(event.target.files?.[0]?.name ?? "选择一个音频文件开始测试");
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
      for (let attempt = 0; attempt < 900; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
        const statusResponse = await fetch(`/api/meeting/${encodeURIComponent(job.task_id)}`);
        const statusPayload = await statusResponse.json() as MeetingStatus | { detail?: string };
        if (!statusResponse.ok) throw new Error("detail" in statusPayload ? statusPayload.detail : "无法读取会议状态");
        const current = statusPayload as MeetingStatus;
        if (current.status === "processing") {
          setStatus(`会议处理中 ${current.elapsed_s ?? attempt + 1}s`);
          continue;
        }
        if (current.status === "failed") throw new Error(current.error ?? "会议处理失败，请查看服务端日志");
        setMeeting({
          task_id: current.task_id,
          filename: file.name,
          duration_s: current.duration_s ?? 0,
          language: "Chinese",
          segments: current.segments ?? 0,
          speakers: current.speakers ?? [],
          report_path: current.report_path ?? job.report_path ?? "",
          summary: current.summary_preview ?? "",
          timeline_preview: current.transcript_preview ?? "",
        });
        setStatus("会议纪要生成完成");
        return;
      }
      throw new Error("会议处理超时，请查看服务端任务目录");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "会议处理失败");
    } finally {
      setBusy(false);
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
