import { useEffect, useState } from "react";

type ReviewStatus = "pending" | "approved" | "rejected";

type ReviewRow = {
  segment_id: string;
  start: number;
  end: number;
  speaker_name?: string;
  speaker?: string;
  raw_text: string;
  corrected_text: string;
  review_status: ReviewStatus;
  audio_url: string;
};

type Pagination = {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
};

type ReviewResponse = {
  items?: ReviewRow[];
  counts?: Record<string, number>;
  pagination?: Pagination;
  detail?: string;
};

const DEFAULT_PAGINATION: Pagination = {
  page: 1,
  page_size: 20,
  total: 0,
  total_pages: 1,
  has_next: false,
  has_prev: false,
};

const formatTime = (seconds: number) => `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

export default function ReviewPage() {
  const [taskId, setTaskId] = useState(() => new URLSearchParams(location.search).get("task_id") ?? localStorage.getItem("astra:lastMeetingTask") ?? "");
  const [rows, setRows] = useState<ReviewRow[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [pagination, setPagination] = useState<Pagination>(DEFAULT_PAGINATION);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [status, setStatus] = useState("输入会议任务 ID，或从会议工作台打开最近任务");
  const [busy, setBusy] = useState(false);

  async function load(targetPage = 1, targetPageSize = pagination.page_size) {
    const normalized = taskId.trim();
    if (!normalized) {
      setStatus("请先输入会议任务 ID");
      return;
    }
    setBusy(true);
    try {
      const query = new URLSearchParams({ page: String(targetPage), page_size: String(targetPageSize) });
      const response = await fetch(`/api/meeting/${encodeURIComponent(normalized)}/training-data?${query}`);
      const payload = await response.json() as ReviewResponse;
      if (!response.ok) throw new Error(payload.detail ?? "无法加载审校数据");
      localStorage.setItem("astra:lastMeetingTask", normalized);
      setRows(payload.items ?? []);
      setCounts(payload.counts ?? {});
      setPagination(payload.pagination ?? { ...DEFAULT_PAGINATION, page: targetPage, page_size: targetPageSize, total: payload.items?.length ?? 0 });
      setSelected(new Set());
      setStatus(`已加载第 ${payload.pagination?.page ?? targetPage} 页，共 ${payload.pagination?.total ?? payload.items?.length ?? 0} 个分段`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "无法加载审校数据");
    } finally {
      setBusy(false);
    }
  }

  async function review(row: ReviewRow, reviewStatus: ReviewStatus) {
    setBusy(true);
    try {
      const response = await fetch(`/api/meeting/${encodeURIComponent(taskId.trim())}/training-data/${encodeURIComponent(row.segment_id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ corrected_text: row.corrected_text.trim(), review_status: reviewStatus }) });
      const payload = await response.json() as { item?: ReviewRow; detail?: string };
      if (!response.ok || !payload.item) throw new Error(payload.detail ?? "保存审校结果失败");
      await load(pagination.page);
      setStatus(`${row.segment_id} 已保存`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "保存审校结果失败");
    } finally {
      setBusy(false);
    }
  }

  async function batchReview(reviewStatus: ReviewStatus) {
    const items = rows
      .filter((row) => selected.has(row.segment_id))
      .map((row) => ({ segment_id: row.segment_id, corrected_text: row.corrected_text.trim() }));
    if (!items.length) {
      setStatus("请先选择当前页的分段");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`/api/meeting/${encodeURIComponent(taskId.trim())}/training-data/batch-review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ review_status: reviewStatus, items }) });
      const payload = await response.json() as { updated_count?: number; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "批量保存审校结果失败");
      await load(pagination.page);
      setStatus(`已批量更新 ${payload.updated_count ?? items.length} 个分段`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "批量保存审校结果失败");
    } finally {
      setBusy(false);
    }
  }

  function toggleRow(segmentId: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(segmentId)) next.delete(segmentId);
      else next.add(segmentId);
      return next;
    });
  }

  const allSelected = rows.length > 0 && rows.every((row) => selected.has(row.segment_id));

  function toggleAll() {
    setSelected((current) => {
      const next = new Set(current);
      if (allSelected) rows.forEach((row) => next.delete(row.segment_id));
      else rows.forEach((row) => next.add(row.segment_id));
      return next;
    });
  }

  useEffect(() => { if (taskId) void load(); }, []);

  return (
    <main className="app-shell upload-shell">
      <header className="topbar">
        <a className="brand" href="/"><span className="brand-mark">A</span><span><strong>Astra</strong><small>Audio Workbench</small></span></a>
        <nav className="nav-actions" aria-label="Astra tools"><a className="nav-link" href="/">实时通话</a><a className="nav-link" href="/upload">会议工作台</a><a className="nav-link active" href="/review">逐段审校</a><a className="nav-link" href="/training">训练设置</a><a className="nav-link" href="/settings">管理设置</a></nav>
      </header>
      <section className="page-heading"><div><span className="eyebrow">TRANSCRIPT REVIEW</span><h1>逐段审校</h1><p>播放原音，确认专业词、人名和数字，审核通过后才进入训练集。</p></div><div className="status-board"><span>当前任务</span><strong>{taskId || "未选择"}</strong><small>{status}</small></div></section>
      <section className="review-toolbar"><label className="training-field training-field-wide"><span>会议任务 ID</span><input value={taskId} onChange={(event) => setTaskId(event.target.value)} placeholder="例如：20260909-175538-74b54122" /></label><button type="button" onClick={() => void load(1)} disabled={busy}>加载审校数据</button><label className="training-field"><span>每页数量</span><select value={pagination.page_size} onChange={(event) => void load(1, Number(event.target.value))} disabled={busy}><option value={10}>10</option><option value={20}>20</option><option value={50}>50</option><option value={100}>100</option></select></label><span>待审核 {counts.pending ?? 0} · 通过 {counts.approved ?? 0} · 驳回 {counts.rejected ?? 0}</span></section>
      {rows.length > 0 && <section className="meeting-review standalone-review"><div className="review-batch-toolbar"><label className="review-select"><input type="checkbox" checked={allSelected} onChange={toggleAll} disabled={busy} />全选当前页</label><span>已选 {selected.size} / {rows.length}</span><button type="button" onClick={() => void batchReview("approved")} disabled={busy || selected.size === 0}>批量通过</button><button type="button" className="secondary-action" onClick={() => void batchReview("pending")} disabled={busy || selected.size === 0}>批量保存待审</button><button type="button" className="danger-action" onClick={() => void batchReview("rejected")} disabled={busy || selected.size === 0}>批量驳回</button></div><div className="meeting-review-list">{rows.map((row) => <div className={`meeting-review-row review-${row.review_status}`} key={row.segment_id}><div className="review-row-meta"><label className="review-select"><input type="checkbox" checked={selected.has(row.segment_id)} onChange={() => toggleRow(row.segment_id)} disabled={busy} aria-label={`选择 ${row.segment_id}`} /></label><strong>{row.segment_id}</strong><span>{formatTime(row.start)} - {formatTime(row.end)} · {row.speaker_name || row.speaker || "说话人待确认"}</span><span className="review-status">{row.review_status === "approved" ? "已通过" : row.review_status === "rejected" ? "已驳回" : "待审核"}</span></div><audio controls preload="none" src={row.audio_url} /><div className="review-text-grid"><div><small>原始识别</small><p>{row.raw_text}</p></div><label><span>确认文本</span><textarea value={row.corrected_text} onChange={(event) => setRows((current) => current.map((item) => item.segment_id === row.segment_id ? { ...item, corrected_text: event.target.value } : item))} /></label></div><div className="review-actions"><button type="button" disabled={busy} onClick={() => void review(row, "approved")}>审核通过</button><button type="button" className="secondary-action" disabled={busy} onClick={() => void review(row, "pending")}>保存待审</button><button type="button" className="danger-action" disabled={busy} onClick={() => void review(row, "rejected")}>驳回</button></div></div>)}</div><div className="review-pagination"><button type="button" disabled={busy || !pagination.has_prev} onClick={() => void load(pagination.page - 1)}>上一页</button><span>第 {pagination.page} / {pagination.total_pages} 页 · 共 {pagination.total} 段</span><button type="button" disabled={busy || !pagination.has_next} onClick={() => void load(pagination.page + 1)}>下一页</button></div></section>}
      {!busy && pagination.total === 0 && taskId.trim() && <p className="empty-state">当前任务没有可审校分段。</p>}
    </main>
  );
}
