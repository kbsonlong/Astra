import { useEffect, useRef, useState } from "react";
import {
  activateVoice,
  addRevision,
  confirmConsent,
  createVoice,
  VoicePreviewController,
  voiceErrorMessage,
} from "./voiceApi";
import type { PreviewAudio, TtsHealth, TtsVoice, VoiceRevision } from "./voice";

const DEFAULT_PREVIEW = "你好，我是 Astra。这是一段本地语音合成试听。";

type Props = {
  existingVoice?: TtsVoice;
  health?: TtsHealth;
  onClose: () => void;
  onComplete: (voice: TtsVoice) => void;
};

function uploadHint(file: File | undefined): string {
  if (!file) return "当前仅支持 WAV；推荐 5–15 秒、单人、安静环境、自然语速。";
  return `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
}

function uploadError(error: unknown): string {
  const generic = voiceErrorMessage(error);
  const detail = error instanceof Error ? error.message : "";
  if (detail.includes("duration")) return "参考音频时长不符合要求：请使用 5–15 秒的单人 WAV 音频。";
  if (detail.includes("mono PCM16")) return "请使用单声道 PCM16 WAV 音频后重试。";
  if (detail.includes("valid WAV")) return "文件不是有效 WAV 音频；请重新导出为 WAV。";
  return generic;
}

export default function VoiceWizard({ existingVoice, health, onClose, onComplete }: Props) {
  const [step, setStep] = useState(1);
  const [name, setName] = useState(existingVoice?.display_name || "");
  const [language, setLanguage] = useState(existingVoice?.revision?.language || "zh");
  const [backendFamily, setBackendFamily] = useState(existingVoice?.revision?.backend_family || "indextts25_mlx");
  const [file, setFile] = useState<File>();
  const [createdVoice, setCreatedVoice] = useState<TtsVoice | undefined>(existingVoice);
  const [revision, setRevision] = useState<VoiceRevision | undefined>(existingVoice?.revision || undefined);
  const [note, setNote] = useState("");
  const [consentChecked, setConsentChecked] = useState(false);
  const [previewText, setPreviewText] = useState(DEFAULT_PREVIEW);
  const [preview, setPreview] = useState<PreviewAudio>();
  const [previewUrl, setPreviewUrl] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const picker = useRef<HTMLInputElement>(null);
  const previewController = useRef(new VoicePreviewController());

  useEffect(() => () => {
    previewController.current.cancel();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  async function submitAudio() {
    if (!file) { setMessage("请选择一个 WAV 参考音频。"); return; }
    if (!file.name.toLowerCase().endsWith(".wav")) { setMessage("当前仅支持 WAV 格式的参考音频。"); return; }
    if (!existingVoice && !name.trim()) { setStep(1); setMessage("请先为音色命名。"); return; }
    setBusy(true); setMessage("");
    try {
      const upload = { file, language, backendFamily };
      if (existingVoice) {
        const newRevision = await addRevision(existingVoice.voice_id, upload);
        setRevision(newRevision);
        setCreatedVoice(await Promise.resolve(existingVoice));
      } else {
        const voice = await createVoice(name.trim(), upload);
        setCreatedVoice(voice);
        setRevision(voice.revision || undefined);
      }
      setStep(3);
    } catch (error) { setMessage(uploadError(error)); } finally { setBusy(false); }
  }

  async function submitConsent() {
    if (!createdVoice || !revision) return;
    setBusy(true); setMessage("");
    try {
      const confirmed = await confirmConsent(createdVoice.voice_id, revision.revision, note.trim() || undefined);
      setRevision(confirmed);
      setStep(4);
    } catch (error) { setMessage(voiceErrorMessage(error)); } finally { setBusy(false); }
  }

  async function makePreview() {
    if (!createdVoice || !revision || !previewText.trim()) return;
    setBusy(true); setMessage("正在生成指定版本的试听…");
    try {
      const result = await previewController.current.request(createdVoice.voice_id, revision.revision, previewText.trim());
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setPreview(result); setPreviewUrl(URL.createObjectURL(result.audio)); setMessage("试听已生成。你可以播放后再决定是否启用。");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setMessage(voiceErrorMessage(error));
    } finally { setBusy(false); }
  }

  async function activate() {
    if (!createdVoice || !revision) return;
    setBusy(true); setMessage("");
    try {
      const voice = await activateVoice(createdVoice.voice_id, revision.revision);
      onComplete(voice);
    } catch (error) { setMessage(voiceErrorMessage(error)); } finally { setBusy(false); }
  }

  const title = existingVoice ? `为“${existingVoice.display_name}”创建新版本` : "新建音色";
  const indexCapability = health?.engines?.indextts_mlx;
  const indexTtsApproved = backendFamily !== "indextts25_mlx" || Boolean(
    indexCapability?.validated && indexCapability.user_enabled && indexCapability.available,
  );
  return <section className="view voice-wizard" aria-label={title}>
    <button className="btn btn--ghost btn--sm voice-detail__back" type="button" onClick={onClose} disabled={busy}>‹ 返回音色库</button>
    <div className="page-head"><div className="page-head__text"><div className="eyebrow">音色向导 · {step} / 4</div><h1>{title}</h1><p>所有参考音频仅用于本机语音合成。创建新版本不会覆盖历史版本或已开始的会话。</p></div></div>
    <ol className="wizard-steps" aria-label="创建进度">{["基本信息", "参考音频", "授权确认", "试听与启用"].map((label, index) => <li className={index + 1 === step ? "wizard-steps__item wizard-steps__item--active" : index + 1 < step ? "wizard-steps__item wizard-steps__item--done" : "wizard-steps__item"} key={label}><span>{index + 1}</span>{label}</li>)}</ol>
    {message && <div className={`alert ${message.includes("已生成") ? "alert--success" : "alert--warning"}`}><span>!</span><div className="alert__body"><div className="alert__desc">{message}</div></div></div>}
    <div className="card wizard-card">
      {step === 1 && <><div className="card__head"><h3>基本信息</h3></div><label className="field"><span className="field__label">音色名称</span><input className="input" value={name} disabled={Boolean(existingVoice)} placeholder="例如：小雅－会议播报" onChange={(event) => setName(event.target.value)} /></label><div className="wizard-fields"><label className="field"><span className="field__label">合成引擎</span><select className="select" value={backendFamily} onChange={(event) => setBackendFamily(event.target.value)}><option value="indextts25_mlx">IndexTTS MLX（实验性）</option><option value="cosyvoice">CosyVoice</option></select>{backendFamily === "indextts25_mlx" && !indexTtsApproved && <span className="field__hint">IndexTTS 尚未完成本机设备验证：可入库并启用，但不能试听或用于实时会话。</span>}</label><label className="field"><span className="field__label">语言</span><select className="select" value={language} onChange={(event) => setLanguage(event.target.value)}><option value="zh">中文</option><option value="en">English</option></select></label></div><div className="wizard-actions"><button className="btn btn--secondary" type="button" onClick={onClose}>取消</button><button className="btn btn--primary" type="button" disabled={!name.trim() && !existingVoice} onClick={() => { setMessage(""); setStep(2); }}>下一步 →</button></div></>}
      {step === 2 && <><div className="card__head"><h3>上传参考音频</h3></div><div className="upload-drop" role="button" tabIndex={0} onClick={() => picker.current?.click()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") picker.current?.click(); }}><strong>选择 WAV 文件</strong><span>{uploadHint(file)}</span><span className="field__hint">请确保录音者已明确授权；不支持 m4a、mp3 或立体声文件。</span></div><input ref={picker} className="sr-only" type="file" accept="audio/wav,.wav" onChange={(event) => { setFile(event.target.files?.[0]); setMessage(""); }} />{file && <div className="upload-file">✓ 已选择：{uploadHint(file)}</div>}<div className="wizard-actions"><button className="btn btn--secondary" type="button" onClick={() => setStep(1)} disabled={busy}>← 上一步</button><button className="btn btn--primary" type="button" onClick={() => void submitAudio()} disabled={!file || busy}>{busy ? "正在保存…" : "下一步 →"}</button></div></>}
      {step === 3 && <><div className="card__head"><h3>使用授权确认</h3></div><div className="consent-copy">该参考录音将用于创建本地合成音色。请确认你拥有录音者的明确授权，或拥有其他足以使用该声音的合法权利。撤销授权后，Astra 会停止新的试听与会话合成请求。</div><label className="check consent-check"><input type="checkbox" checked={consentChecked} onChange={(event) => setConsentChecked(event.target.checked)} />我确认拥有使用此录音创建和使用合成音色的授权。</label><label className="field"><span className="field__label">授权备注 <span className="field__hint">可选</span></span><textarea className="textarea" value={note} maxLength={1000} placeholder="例如：本人录制，仅用于内部会议播报" onChange={(event) => setNote(event.target.value)} /></label><div className="wizard-actions"><button className="btn btn--secondary" type="button" onClick={() => setStep(2)} disabled={busy}>← 上一步</button><button className="btn btn--primary" type="button" onClick={() => void submitConsent()} disabled={busy || !consentChecked}>{busy ? "正在确认…" : "我已确认授权 →"}</button></div></>}
      {step === 4 && <><div className="card__head"><h3>试听与启用</h3><span className="card__hint">r{revision?.revision} · 已确认授权</span></div><label className="field"><span className="field__label">试听文案 <span className="field__hint">{previewText.length} / 200</span></span><textarea className="textarea" value={previewText} maxLength={200} onChange={(event) => setPreviewText(event.target.value)} /></label>{!indexTtsApproved && <div className="alert alert--warning"><span>!</span><div className="alert__body"><div className="alert__title">IndexTTS 等待设备验证</div><div className="alert__desc">模型存在不等于可向用户开放。完成目标设备的 L2/L3 验证后，管理员可设置 TTS_INDEXTTS_VALIDATED=true。</div></div></div>}<button className="btn btn--secondary" type="button" onClick={() => void makePreview()} disabled={busy || !previewText.trim() || !indexTtsApproved}>{busy ? "正在生成…" : "▶ 生成试听"}</button>{previewUrl && <audio className="wizard-preview" controls autoPlay src={previewUrl}>你的浏览器不支持音频播放。</audio>}{preview && <p className="card__hint">{preview.backend || "当前引擎"}{preview.modelRevision ? ` · 模型 ${preview.modelRevision}` : ""}</p>}<div className="wizard-actions"><button className="btn btn--ghost" type="button" onClick={onClose} disabled={busy}>保留为草稿</button><button className="btn btn--primary" type="button" onClick={() => void activate()} disabled={busy}>{busy ? "正在启用…" : "启用音色"}</button></div><p className="field__hint">试听是推荐的预检，不是启用的前置条件；模型暂不可用时仍可启用，实时会话会在引擎就绪后才能使用。</p></>}
    </div>
  </section>;
}
