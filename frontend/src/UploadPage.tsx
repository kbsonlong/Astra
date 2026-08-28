import { ChangeEvent, FormEvent, useState } from "react";

type TranscriptionResult = {
  filename: string;
  bytes: number;
  text: string;
};

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [status, setStatus] = useState("选择一个音频文件开始测试");
  const [busy, setBusy] = useState(false);

  function selectFile(event: ChangeEvent<HTMLInputElement>) {
    setFile(event.target.files?.[0] ?? null);
    setResult(null);
    setStatus(event.target.files?.[0]?.name ?? "选择一个音频文件开始测试");
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
        <button type="submit" disabled={busy || !file}>{busy ? "转写中..." : "POST 转写"}</button>
      </form>
      {result && (
        <section className="result" aria-live="polite">
          <span className="eyebrow">TRANSCRIPTION RESULT</span>
          <p className="result-meta">{result.filename} · {result.bytes.toLocaleString()} bytes</p>
          <p className="result-text">{result.text || "未识别到文本"}</p>
        </section>
      )}
    </main>
  );
}
