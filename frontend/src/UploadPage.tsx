import { ChangeEvent, FormEvent, useState } from "react";

type TranscriptionResult = {
  filename: string;
  bytes: number;
  text: string;
};

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

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [correction, setCorrection] = useState("");
  const [status, setStatus] = useState("选择一个音频文件开始测试");
  const [busy, setBusy] = useState(false);
  const [topic, setTopic] = useState("");
  const [meeting, setMeeting] = useState<MeetingResult | null>(null);

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setResult(null);
    setCorrection("");
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
          const payload = JSON.parse(data) as { type: string; text?: string; token?: string; message?: string };
          if (payload.type === "asr_final") setResult({ filename: file.name, bytes: file.size, text: payload.text ?? "" });
          if (payload.type === "correction_token") setCorrection((current) => current + (payload.token ?? ""));
          if (payload.type === "correction_final") setCorrection(payload.text ?? "");
          if (payload.type === "error") throw new Error(payload.message ?? "流式修正失败");
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
      setMeeting(payload as MeetingResult);
      setStatus("会议纪要生成完成");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "会议处理失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell upload-shell">
      <header>
        <span className="eyebrow">ASTRA / ASR TEST</span>
        <h1>音频上传</h1>
        <a className="back-link" href="/">返回通话</a>
      </header>
      <form className="upload-form" onSubmit={upload}>
        <label className="file-picker">
          <span>选择音频文件</span>
          <input type="file" accept="audio/*,.wav" onChange={selectFile} />
        </label>
        <p className="upload-status">{status}</p>
        <div className="upload-actions">
          <button type="submit" disabled={busy || !file}>{busy ? "转写中..." : "POST 转写"}</button>
          <button type="button" disabled={busy || !file} onClick={uploadWithCorrection}>流式修正</button>
          <button type="button" className="meeting-btn" disabled={busy || !file} onClick={runMeeting}>
            {busy ? "处理中..." : "生成会议纪要"}
          </button>
        </div>
        <label className="meeting-topic">
          <span>会议主题（可选，辅助摘要）</span>
          <input type="text" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="例如：云资源成本优化周会" />
        </label>
      </form>
      {meeting && (
        <section className="result meeting-result" aria-live="polite">
          <span className="eyebrow">MEETING NOTES · {meeting.task_id}</span>
          <p className="result-meta">
            {meeting.filename} · {Math.round(meeting.duration_s / 60)}分{Math.round(meeting.duration_s % 60)}秒 ·
            说话人 {meeting.speakers.length > 0 ? meeting.speakers.join("/") : "未分离"} · {meeting.segments} 段
          </p>
          {meeting.summary && <p className="result-text summary-text">{meeting.summary}</p>}
          {meeting.timeline_preview && (
            <details open={false}>
              <summary>逐字稿预览</summary>
              <pre className="timeline-pre">{meeting.timeline_preview}</pre>
            </details>
          )}
          <p className="result-meta">完整报告: {meeting.report_path}</p>
        </section>
      )}
      {result && (
        <section className="result" aria-live="polite">
          <span className="eyebrow">TRANSCRIPTION RESULT</span>
          <p className="result-meta">{result.filename} · {result.bytes.toLocaleString()} bytes</p>
          <p className="result-text">{result.text || "未识别到文本"}</p>
        </section>
      )}
      {correction && (
        <section className="result correction-result" aria-live="polite">
          <span className="eyebrow">LLM CORRECTION STREAM</span>
          <p className="result-text">{correction}</p>
        </section>
      )}
    </main>
  );
}
