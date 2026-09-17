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

  const statusLabel = (s: ReviewStatus) =>
    s === "approved" ? "已通过" : s === "rejected" ? "已驳回" : "待审核";
  const statusBadge = (s: ReviewStatus) =>
    s === "approved" ? "badge--success" : s === "rejected" ? "badge--danger" : "badge--warning";

  return (
    <section className="view" aria-label="会议复核">
      <div className="page-head">
        <div className="page-head__text">
          <div className="eyebrow">03 · 会议复核</div>
          <h1>逐段复核 · {taskId || "未选择任务"}</h1>
          <p>播放原音，核对说话人与纠错结果。审核通过的数据才能用于 ASR 微调。</p>
        </div>
        <div className="page-head__actions">
          <span className="badge badge--warning">待审核 {counts.pending ?? 0}</span>
          <span className="badge badge--success">通过 {counts.approved ?? 0}</span>
          <span className="badge badge--danger">驳回 {counts.rejected ?? 0}</span>
        </div>
      </div>

      <div className="card" style={{ marginBottom: "var(--space-5)" }}>
        <div className="form-grid">
          <div className="field">
            <label className="field__label" htmlFor="rv-task">会议任务 ID</label>
            <input
              className="input input--mono"
              id="rv-task"
              value={taskId}
              onChange={(event) => setTaskId(event.target.value)}
              placeholder="例如：20260909-175538-74b54122"
            />
          </div>
          <div className="field">
            <label className="field__label" htmlFor="rv-size">每页数量</label>
            <select
              className="select"
              id="rv-size"
              value={pagination.page_size}
              onChange={(event) => void load(1, Number(event.target.value))}
              disabled={busy}
            >
              <option value={10}>10</option>
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </div>
          <div className="field" style={{ justifyContent: "flex-end" }}>
            <button className="btn btn--primary" type="button" onClick={() => void load(1)} disabled={busy}>
              加载审校数据
            </button>
          </div>
        </div>
        <p className="card__hint" style={{ marginTop: "var(--space-3)" }}>{status}</p>
      </div>

      {rows.length > 0 && (
        <div className="card">
          <div className="review-batch">
            <label className="check">
              <input type="checkbox" checked={allSelected} onChange={toggleAll} disabled={busy} />
              全选当前页
            </label>
            <span className="review-batch__count">已选 {selected.size} / {rows.length}</span>
            <button className="btn btn--primary btn--sm" type="button" onClick={() => void batchReview("approved")} disabled={busy || selected.size === 0}>批量通过</button>
            <button className="btn btn--secondary btn--sm" type="button" onClick={() => void batchReview("pending")} disabled={busy || selected.size === 0}>批量保存待审</button>
            <button className="btn btn--danger btn--sm" type="button" onClick={() => void batchReview("rejected")} disabled={busy || selected.size === 0}>批量驳回</button>
          </div>
          <div className="divider" />
          <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-5)" }}>
            {rows.map((row) => (
              <div key={row.segment_id}>
                <div className="card__head">
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={selected.has(row.segment_id)}
                      onChange={() => toggleRow(row.segment_id)}
                      disabled={busy}
                      aria-label={`选择 ${row.segment_id}`}
                    />
                  </label>
                  <div style={{ minWidth: 0 }}>
                    <div className="filecard__name">{row.segment_id}</div>
                    <div className="filecard__meta">
                      {formatTime(row.start)} – {formatTime(row.end)} · {row.speaker_name || row.speaker || "说话人待确认"}
                    </div>
                  </div>
                  <span className={`badge ${statusBadge(row.review_status)}`} style={{ marginLeft: "auto" }}>
                    {statusLabel(row.review_status)}
                  </span>
                </div>
                <div className="player" style={{ marginBottom: "var(--space-3)" }}>
                  <audio controls preload="none" src={row.audio_url} />
                </div>
                <div className="emend" style={{ marginBottom: "var(--space-3)" }}>
                  <span className="emend__orig">{row.raw_text}</span>
                </div>
                <div className="field">
                  <label className="field__label" htmlFor={`rv-edit-${row.segment_id}`}>确认文本</label>
                  <textarea
                    className="textarea"
                    id={`rv-edit-${row.segment_id}`}
                    value={row.corrected_text}
                    onChange={(event) =>
                      setRows((current) =>
                        current.map((item) =>
                          item.segment_id === row.segment_id
                            ? { ...item, corrected_text: event.target.value }
                            : item,
                        ),
                      )
                    }
                  />
                </div>
                <div className="review-batch" style={{ paddingBottom: 0 }}>
                  <button className="btn btn--primary btn--sm" type="button" disabled={busy} onClick={() => void review(row, "approved")}>审核通过</button>
                  <button className="btn btn--secondary btn--sm" type="button" disabled={busy} onClick={() => void review(row, "pending")}>保存待审</button>
                  <button className="btn btn--danger btn--sm" type="button" disabled={busy} onClick={() => void review(row, "rejected")}>驳回</button>
                </div>
                <div className="divider" />
              </div>
            ))}
          </div>
          <div className="pager">
            <button className="btn btn--secondary btn--sm" type="button" disabled={busy || !pagination.has_prev} onClick={() => void load(pagination.page - 1)}>上一页</button>
            <span className="card__hint">第 {pagination.page} / {pagination.total_pages} 页 · 共 {pagination.total} 段</span>
            <button className="btn btn--secondary btn--sm" type="button" disabled={busy || !pagination.has_next} onClick={() => void load(pagination.page + 1)}>下一页</button>
          </div>
        </div>
      )}

      {!busy && pagination.total === 0 && taskId.trim() && (
        <div className="empty">
          <div className="empty__icon">◎</div>
          <div className="empty__title">当前任务没有可审校分段</div>
          <div className="empty__desc">确认任务 ID 是否正确，或从音频工作台打开最近完成的会议任务。</div>
        </div>
      )}
    </section>
  );
}
