import { useEffect, useState } from "react";

type ReviewRow = {
  segment_id: string;
  start: number;
  end: number;
  speaker_name?: string;
  speaker?: string;
  raw_text: string;
  corrected_text: string;
  review_status: "pending" | "approved" | "rejected";
  audio_url: string;
};

const formatTime = (seconds: number) => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

export default function ReviewPage() {
  const [taskId, setTaskId] = useState(() => new URLSearchParams(location.search).get("task_id") ?? localStorage.getItem("astra:lastMeetingTask") ?? "");
  const [rows, setRows] = useState<ReviewRow[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [status, setStatus] = useState("输入会议任务 ID，或从会议工作台打开最近任务");
  const [busy, setBusy] = useState(false);

  async function load() {
    const normalized = taskId.trim();
    if (!normalized) {
      setStatus("请先输入会议任务 ID");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`/api/meeting/${encodeURIComponent(normalized)}/training-data`);
      const payload = await response.json() as { items?: ReviewRow[]; counts?: Record<string, number>; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "无法加载审校数据");
      localStorage.setItem("astra:lastMeetingTask", normalized);
      setRows(payload.items ?? []);
      setCounts(payload.counts ?? {});
      setStatus(`已加载 ${payload.items?.length ?? 0} 个分段`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "无法加载审校数据");
    } finally {
      setBusy(false);
    }
  }

  async function review(row: ReviewRow, reviewStatus: ReviewRow["review_status"]) {
    setBusy(true);
    try {
      const response = await fetch(`/api/meeting/${encodeURIComponent(taskId)}/training-data/${encodeURIComponent(row.segment_id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ corrected_text: row.corrected_text.trim(), review_status: reviewStatus }) });
      const payload = await response.json() as { item?: ReviewRow; detail?: string };
      if (!response.ok || !payload.item) throw new Error(payload.detail ?? "保存审校结果失败");
      setRows((current) => current.map((item) => item.segment_id === row.segment_id ? payload.item! : item));
      await load();
      setStatus(`${row.segment_id} 已保存`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "保存审校结果失败");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => { if (taskId) void load(); }, []);

  return (
    <main className="app-shell upload-shell">
      <header className="topbar">
        <a className="brand" href="/"><span className="brand-mark">A</span><span><strong>Astra</strong><small>Audio Workbench</small></span></a>
        <nav className="nav-actions" aria-label="Astra tools"><a className="nav-link" href="/">实时通话</a><a className="nav-link" href="/upload">会议工作台</a><a className="nav-link active" href="/review">逐段审校</a><a className="nav-link" href="/training">训练设置</a></nav>
      </header>
      <section className="page-heading"><div><span className="eyebrow">TRANSCRIPT REVIEW</span><h1>逐段审校</h1><p>播放原音，确认专业词、人名和数字，审核通过后才进入训练集。</p></div><div className="status-board"><span>当前任务</span><strong>{taskId || "未选择"}</strong><small>{status}</small></div></section>
      <section className="review-toolbar"><label className="training-field training-field-wide"><span>会议任务 ID</span><input value={taskId} onChange={(event) => setTaskId(event.target.value)} placeholder="例如：20260909-175538-74b54122" /></label><button type="button" onClick={() => void load()} disabled={busy}>加载审校数据</button><span>待审核 {counts.pending ?? 0} · 通过 {counts.approved ?? 0} · 驳回 {counts.rejected ?? 0}</span></section>
      {rows.length > 0 && <section className="meeting-review standalone-review"><div className="meeting-review-list">{rows.map((row) => <div className={`meeting-review-row review-${row.review_status}`} key={row.segment_id}><div className="review-row-meta"><strong>{row.segment_id}</strong><span>{formatTime(row.start)} - {formatTime(row.end)} · {row.speaker_name || row.speaker || "说话人待确认"}</span><span className="review-status">{row.review_status === "approved" ? "已通过" : row.review_status === "rejected" ? "已驳回" : "待审核"}</span></div><audio controls preload="none" src={row.audio_url} /><div className="review-text-grid"><div><small>原始识别</small><p>{row.raw_text}</p></div><label><span>确认文本</span><textarea value={row.corrected_text} onChange={(event) => setRows((current) => current.map((item) => item.segment_id === row.segment_id ? { ...item, corrected_text: event.target.value } : item))} /></label></div><div className="review-actions"><button type="button" disabled={busy} onClick={() => void review(row, "approved")}>审核通过</button><button type="button" className="secondary-action" disabled={busy} onClick={() => void review(row, "pending")}>保存待审</button><button type="button" className="danger-action" disabled={busy} onClick={() => void review(row, "rejected")}>驳回</button></div></div>)}</div></section>}
    </main>
  );
}
