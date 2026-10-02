import { useEffect, useMemo, useState } from "react";
import {
  archiveVoice,
  getTtsHealth,
  getVoice,
  listRevisions,
  listVoices,
  restoreVoice,
  revokeConsent,
  voiceErrorMessage,
} from "./voiceApi";
import { availabilityLabel, voiceAvailability } from "./voiceState";
import type { TtsHealth, TtsVoice, VoiceListFilters, VoiceRevision } from "./voice";
import VoiceWizard from "./VoiceWizard";

type DetailState = { voice: TtsVoice; revisions: VoiceRevision[] };

function dateLabel(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
}

function durationLabel(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function statusBadge(status: string): { label: string; style: string } {
  switch (status) {
    case "active": return { label: "可用", style: "badge--success" };
    case "draft": return { label: "草稿", style: "badge--warning" };
    case "archived": return { label: "已归档", style: "badge--neutral" };
    case "revoked": return { label: "已撤销", style: "badge--danger" };
    default: return { label: status, style: "badge--neutral" };
  }
}

function backendLabel(backend: string): string {
  if (backend === "indextts25_mlx" || backend === "indextts_mlx") return "IndexTTS MLX";
  if (backend === "cosyvoice") return "CosyVoice";
  return backend || "未指定引擎";
}

export default function VoiceLibraryPage() {
  const [filters, setFilters] = useState<VoiceListFilters>({ page: 1, pageSize: 24 });
  const [queryDraft, setQueryDraft] = useState("");
  const [voices, setVoices] = useState<TtsVoice[]>([]);
  const [total, setTotal] = useState(0);
  const [health, setHealth] = useState<TtsHealth>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [detail, setDetail] = useState<DetailState>();
  const [wizardFor, setWizardFor] = useState<TtsVoice | null | undefined>(undefined);

  useEffect(() => {
    const timer = window.setTimeout(() => setFilters((value) => ({ ...value, query: queryDraft || undefined, page: 1 })), 220);
    return () => window.clearTimeout(timer);
  }, [queryDraft]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void Promise.all([listVoices(filters, { signal: controller.signal }), getTtsHealth({ signal: controller.signal })])
      .then(([result, tts]) => {
        setVoices(result.items);
        setTotal(result.pagination.total);
        setHealth(tts);
      })
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(voiceErrorMessage(reason));
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [filters]);

  const counts = useMemo(() => ({
    active: voices.filter((voice) => voice.status === "active").length,
    pending: voices.filter((voice) => voice.revision?.consent_status !== "confirmed" && voice.status !== "revoked").length,
  }), [voices]);
  const currentPage = filters.page || 1;
  const pageSize = filters.pageSize || 24;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));

  async function openDetail(voiceId: string) {
    setError("");
    try {
      const [voice, revisions] = await Promise.all([getVoice(voiceId), listRevisions(voiceId)]);
      setDetail({ voice, revisions: revisions.items });
    } catch (reason) {
      setError(voiceErrorMessage(reason));
    }
  }

  function updateFilter<K extends keyof VoiceListFilters>(key: K, value: VoiceListFilters[K]) {
    setFilters((current) => ({ ...current, [key]: value || undefined, page: 1 }));
  }

  if (detail) {
    return <VoiceDetail detail={detail} health={health} onBack={() => setDetail(undefined)} onNewRevision={() => setWizardFor(detail.voice)} onChanged={() => void openDetail(detail.voice.voice_id)} />;
  }
  if (wizardFor !== undefined) {
    return <VoiceWizard existingVoice={wizardFor || undefined} health={health} onClose={() => setWizardFor(undefined)} onComplete={(voice) => { setWizardFor(undefined); void openDetail(voice.voice_id); setFilters((current) => ({ ...current })); }} />;
  }

  return (
    <section className="view voice-library" aria-label="音色库">
      <div className="page-head">
        <div className="page-head__text">
          <div className="eyebrow">05 · 音色库</div>
          <h1>本地音色库</h1>
          <p>管理用于语音合成的参考音色。会议说话人档案仅用于识别，不会自动用于语音克隆。</p>
        </div>
        <div className="page-head__actions">
          <button className="btn btn--primary" type="button" onClick={() => setWizardFor(null)}>＋ 新建音色</button>
        </div>
      </div>

      {!health?.available && health && (
        <div className="alert alert--warning voice-library__engine-alert">
          <span>!</span><div className="alert__body"><div className="alert__title">TTS 当前不可用</div><div className="alert__desc">{health.reason || "你仍可管理音色资料，但不能试听或用于实时会话。"}</div></div>
        </div>
      )}
      {error && <div className="alert alert--danger"><span>!</span><div className="alert__body"><div className="alert__title">无法加载音色库</div><div className="alert__desc">{error}</div></div></div>}

      <div className="voice-toolbar card">
        <label className="voice-search"><span className="sr-only">搜索音色</span><input className="input" value={queryDraft} placeholder="搜索音色名称" onChange={(event) => setQueryDraft(event.target.value)} /></label>
        <select className="select" value={filters.status || ""} onChange={(event) => updateFilter("status", event.target.value)} aria-label="按状态筛选">
          <option value="">全部状态</option><option value="active">可用</option><option value="draft">草稿</option><option value="archived">已归档</option><option value="revoked">已撤销</option>
        </select>
        <select className="select" value={filters.backendFamily || ""} onChange={(event) => updateFilter("backendFamily", event.target.value)} aria-label="按引擎筛选">
          <option value="">全部引擎</option><option value="indextts25_mlx">IndexTTS MLX</option><option value="cosyvoice">CosyVoice</option>
        </select>
        <div className="voice-toolbar__summary" aria-live="polite">可用 {counts.active} · 待授权 {counts.pending} · 共 {total}</div>
      </div>

      {loading ? <LoadingCards /> : voices.length === 0 ? <EmptyLibrary onCreate={() => setWizardFor(null)} /> : (
        <>
          <div className="voice-grid">
            {voices.map((voice) => <VoiceCard key={voice.voice_id} voice={voice} health={health} onOpen={() => void openDetail(voice.voice_id)} />)}
          </div>
          {pageCount > 1 && <nav className="voice-pagination" aria-label="音色库分页"><button className="btn btn--secondary btn--sm" type="button" disabled={currentPage === 1} onClick={() => setFilters((current) => ({ ...current, page: currentPage - 1 }))}>上一页</button><span>第 {currentPage} / {pageCount} 页</span><button className="btn btn--secondary btn--sm" type="button" disabled={currentPage === pageCount} onClick={() => setFilters((current) => ({ ...current, page: currentPage + 1 }))}>下一页</button></nav>}
        </>
      )}
    </section>
  );
}

function VoiceCard({ voice, health, onOpen }: { voice: TtsVoice; health?: TtsHealth; onOpen: () => void }) {
  const revision = voice.revision;
  const badge = statusBadge(voice.status);
  const availability = revision ? voiceAvailability(revision, health) : undefined;
  return <article className="voice-card" tabIndex={0} role="button" onClick={onOpen} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") onOpen(); }}>
    <div className="voice-card__head"><div className="voice-avatar" aria-hidden="true">◉</div><div><h2>{voice.display_name}</h2><p>{revision ? `${backendLabel(revision.backend_family)} · ${revision.language} · r${revision.revision}` : "尚未创建版本"}</p></div><span className={`badge ${badge.style}`}>{badge.label}</span></div>
    {revision && <>
      <div className="voice-card__meta">参考音频 {durationLabel(revision.duration_s)} · {revision.sample_rate / 1000} kHz</div>
      <div className="voice-card__status">{availabilityLabel(availability!)}</div>
    </>}
    <div className="voice-card__foot"><span>更新于 {dateLabel(voice.updated_at)}</span><button type="button" className="btn btn--ghost btn--sm" onClick={(event) => { event.stopPropagation(); onOpen(); }}>查看详情</button></div>
  </article>;
}

function VoiceDetail({ detail, health, onBack, onNewRevision, onChanged }: { detail: DetailState; health?: TtsHealth; onBack: () => void; onNewRevision: () => void; onChanged: () => void }) {
  const { voice, revisions } = detail;
  const revision = voice.revision;
  const availability = revision ? voiceAvailability(revision, health) : undefined;
  const badge = statusBadge(voice.status);
  return <section className="view voice-detail" aria-label={`${voice.display_name}详情`}>
    <button className="btn btn--ghost btn--sm voice-detail__back" type="button" onClick={onBack}>‹ 返回音色库</button>
    <div className="page-head"><div className="page-head__text"><div className="eyebrow">音色详情</div><h1>{voice.display_name}</h1><p>{revision ? `${backendLabel(revision.backend_family)} · ${revision.language} · 当前 r${revision.revision}` : "尚无版本"}</p></div><div className="page-head__actions"><span className={`badge ${badge.style}`}>{badge.label}</span>{availability?.canUseInSession && <button className="btn btn--primary" type="button" onClick={() => window.location.assign(`/?voice_id=${encodeURIComponent(voice.voice_id)}&voice_revision=${revision!.revision}`)}>用于本次会话</button>}<button className="btn btn--secondary" type="button" onClick={onNewRevision}>创建新版本</button><VoiceRiskActions voice={voice} revision={revision} onChanged={onChanged} /></div></div>
    {revision ? <div className="voice-detail__grid">
      <div className="voice-detail__main">
        <section className="card"><div className="card__head"><h3>当前版本 r{revision.revision}</h3><span className="card__hint">{availabilityLabel(availability!)}</span></div><div className="reference-audio"><div><strong>参考音频</strong><span>{durationLabel(revision.duration_s)} · {revision.sample_rate / 1000} kHz · 单声道</span></div>{availability!.canPlayReference ? <audio controls preload="metadata" src={revision.audio_url}>你的浏览器不支持音频播放。</audio> : <p>授权已撤销，参考音频不可播放。</p>}</div><dl className="voice-facts"><div><dt>引擎</dt><dd>{backendLabel(revision.backend_family)}</dd></div><div><dt>SHA-256</dt><dd title={revision.reference_sha256}>{revision.reference_sha256.slice(0, 12)}…</dd></div><div><dt>创建时间</dt><dd>{dateLabel(revision.created_at)}</dd></div></dl></section>
        <section className="card"><div className="card__head"><h3>授权状态</h3><span className={`badge ${revision.consent_status === "confirmed" ? "badge--success" : revision.consent_status === "revoked" ? "badge--danger" : "badge--warning"}`}>{revision.consent_status === "confirmed" ? "已确认授权" : revision.consent_status === "revoked" ? "已撤销" : "待确认"}</span></div><p className="card__hint">{revision.consent_status === "confirmed" ? `已于 ${dateLabel(revision.consent_confirmed_at)} 确认。` : "确认授权与撤销授权将在新建音色流程中完成。"}</p></section>
        <section className="card"><div className="card__head"><h3>版本历史</h3><span className="card__hint">历史会话会固定引用创建时的版本</span></div><div className="revision-list">{revisions.map((item) => <div className="revision-row" key={item.revision}><strong>r{item.revision}</strong><span>{item.revision === voice.current_revision ? "当前版本" : "历史版本"}</span><span>{backendLabel(item.backend_family)} · {durationLabel(item.duration_s)}</span><span>{item.consent_status === "confirmed" ? "已确认授权" : item.consent_status === "revoked" ? "已撤销" : "待授权"}</span></div>)}</div></section>
      </div>
      <aside className="card preview-card"><div className="card__head"><h3>试听合成</h3></div><p className="card__hint">可在“创建新版本”向导中为指定版本生成试听。当前状态：{availabilityLabel(availability!)}。</p><button className="btn btn--secondary btn--block" type="button" onClick={onNewRevision}>创建新版本并试听</button></aside>
    </div> : <EmptyLibrary />}
  </section>;
}

function VoiceRiskActions({ voice, revision, onChanged }: { voice: TtsVoice; revision: VoiceRevision | null; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  async function run(action: "archive" | "restore" | "revoke") {
    if (action === "archive" && !window.confirm("归档不会删除音色，但会使其暂时不可用于新会话。确定归档吗？")) return;
    if (action === "revoke" && !window.confirm("撤销授权会立即阻断新的试听与会话合成，且不能自动恢复。确定继续吗？")) return;
    setBusy(true); setMessage("");
    try {
      if (action === "archive") await archiveVoice(voice.voice_id);
      else if (action === "restore") await restoreVoice(voice.voice_id);
      else if (revision) await revokeConsent(voice.voice_id, revision.revision, window.prompt("撤销原因（可选）") || undefined);
      onChanged();
    } catch (error) { setMessage(voiceErrorMessage(error)); } finally { setBusy(false); }
  }
  return <><button className="btn btn--ghost" type="button" disabled={busy || voice.status === "revoked"} onClick={() => void run(voice.status === "archived" ? "restore" : "archive")}>{voice.status === "archived" ? "恢复归档" : "归档"}</button>{revision?.consent_status === "confirmed" && voice.status !== "revoked" && <button className="btn btn--danger" type="button" disabled={busy} onClick={() => void run("revoke")}>撤销授权</button>}{message && <span className="field__error">{message}</span>}</>;
}

function LoadingCards() { return <div className="voice-grid" aria-label="正在加载音色">{[1, 2, 3].map((item) => <div className="voice-card voice-card--loading" key={item}><div className="skeleton skeleton--title" /><div className="skeleton" /><div className="skeleton" /></div>)}</div>; }
function EmptyLibrary({ onCreate }: { onCreate?: () => void }) { return <div className="empty"><div className="empty__icon">◌</div><div className="empty__title">还没有符合条件的合成音色</div><div className="empty__desc">上传一段已获授权的单人 WAV 参考音频后，即可在此查看状态、版本和试听效果。</div>{onCreate && <button className="btn btn--secondary" type="button" onClick={onCreate}>新建第一个音色</button>}</div>; }
