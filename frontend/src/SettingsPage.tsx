import { useEffect, useState } from "react";

type LlmConfig = {
  llm_base_url: string;
  llm_chat_path: string;
  llm_models_path: string;
  llm_model: string;
  llm_api_key: string;
  llm_api_key_configured: boolean;
  llm_request_timeout_seconds: number;
  llm_connect_timeout_seconds: number;
  llm_stream_idle_timeout_seconds: number;
  llm_correction_enabled: boolean;
  llm_correction_max_tokens: number;
  llm_correction_system_prompt: string;
  meeting_llm_correction_enabled: boolean;
  meeting_llm_correction_candidates: string[];
};

type ApiPayload = LlmConfig & {
  detail?: unknown;
  saved?: boolean;
  runtime_applied?: boolean;
};

function errorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== "object") return String(item);
      const entry = item as { loc?: unknown[]; msg?: unknown };
      const location = Array.isArray(entry.loc) ? entry.loc.join(".") : "字段";
      return `${location}: ${String(entry.msg ?? "参数无效")}`;
    });
    if (messages.length) return messages.join("；");
  }
  return fallback;
}

export default function SettingsPage() {
  const [config, setConfig] = useState<LlmConfig | null>(null);
  const [apiKeyDraft, setApiKeyDraft] = useState("");
  const [clearApiKey, setClearApiKey] = useState(false);
  const [status, setStatus] = useState("正在读取 LLM 配置…");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void loadConfig();
  }, []);

  async function loadConfig() {
    setBusy(true);
    try {
      const response = await fetch("/api/config");
      const payload = await response.json() as ApiPayload;
      if (!response.ok) throw new Error(errorMessage(payload.detail, "无法读取 LLM 配置"));
      setConfig(payload);
      setApiKeyDraft("");
      setClearApiKey(false);
      setStatus(payload.llm_api_key_configured ? "已加载，API Key 已隐藏" : "已加载，尚未配置 API Key");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "LLM 配置读取失败");
    } finally {
      setBusy(false);
    }
  }

  async function saveConfig() {
    if (!config) return;
    setBusy(true);
    try {
      const response = await fetch("/api/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...config,
          llm_api_key: clearApiKey ? "" : apiKeyDraft.trim() || null,
        }),
      });
      const payload = await response.json() as ApiPayload;
      if (!response.ok || !payload.llm_base_url) throw new Error(errorMessage(payload.detail, "LLM 配置保存失败"));
      setConfig(payload);
      setApiKeyDraft("");
      setClearApiKey(false);
      setStatus("已保存并立即应用到实时对话和会议处理");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "LLM 配置保存失败");
    } finally {
      setBusy(false);
    }
  }

  async function checkHealth() {
    setBusy(true);
    try {
      const response = await fetch("/api/health");
      const payload = await response.json() as { llm?: { ok?: boolean; detail?: string }; detail?: string };
      if (!response.ok) throw new Error(errorMessage(payload.detail, "健康检查失败"));
      if (payload.llm?.ok) setStatus("LLM 连接正常");
      else setStatus(`LLM 连接失败${payload.llm?.detail ? `：${payload.llm.detail}` : ""}`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "LLM 健康检查失败");
    } finally {
      setBusy(false);
    }
  }

  function update<K extends keyof LlmConfig>(key: K, value: LlmConfig[K]) {
    setConfig((current) => current ? { ...current, [key]: value } : current);
  }

  function textField(key: "llm_base_url" | "llm_model" | "llm_chat_path" | "llm_models_path", label: string, hint?: string) {
    if (!config) return null;
    return <label className="training-field settings-field"><span>{label}</span><input value={config[key]} onChange={(event) => update(key, event.target.value)} />{hint && <small className="field-hint">{hint}</small>}</label>;
  }

  function numberField(key: "llm_request_timeout_seconds" | "llm_connect_timeout_seconds" | "llm_stream_idle_timeout_seconds" | "llm_correction_max_tokens", label: string, step = "1") {
    if (!config) return null;
    return <label className="training-field settings-field"><span>{label}</span><input type="number" min="1" step={step} value={config[key]} onChange={(event) => update(key, Number(event.target.value) as LlmConfig[typeof key])} /></label>;
  }

  return (
    <main className="app-shell upload-shell">
      <header className="topbar">
        <a className="brand" href="/"><span className="brand-mark">A</span><span><strong>Astra</strong><small>Local Voice Assistant</small></span></a>
        <nav className="nav-actions" aria-label="Astra tools">
          <a className="nav-link" href="/">实时通话</a>
          <a className="nav-link" href="/upload">会议工作台</a>
          <a className="nav-link" href="/review">逐段审校</a>
          <a className="nav-link" href="/training">训练设置</a>
          <a className="nav-link active" href="/settings">管理设置</a>
        </nav>
      </header>

      <section className="page-heading">
        <div><span className="eyebrow">LLM CONNECTION · ENV MANAGEMENT</span><h1>管理设置</h1><p>在这里配置 LLM 连接参数。保存后会写入 `.env`，并立即应用到实时对话和会议纪要。</p></div>
        <div className="status-board"><span>当前状态</span><strong>{config?.llm_api_key_configured ? "已配置" : "待配置"}</strong><small>{status}</small></div>
      </section>

      <section className="training-panel settings-panel">
        <div className="speaker-panel-head"><div><h2>LLM 连接</h2><p className="training-note">支持本地或远程 OpenAI 兼容服务；只需填写服务地址和模型名即可开始。</p></div><div className="speaker-actions"><button type="button" className="secondary-action compact-button" onClick={() => void loadConfig()} disabled={busy}>重载</button><button type="button" className="compact-button" onClick={() => void checkHealth()} disabled={!config || busy}>测试连接</button><button type="button" className="compact-button" onClick={() => void saveConfig()} disabled={!config || busy}>保存并应用</button></div></div>
        <p className="speaker-status">{status}</p>
        {config && <>
          <div className="settings-section-title">连接参数</div>
          <div className="training-grid settings-grid">
            {textField("llm_base_url", "服务地址", "例如 http://127.0.0.1:8000/v1，也可以填写局域网或远程地址。")}
            {textField("llm_model", "模型名称", "必须与 LLM 服务实际提供的模型 ID 一致。")}
            {textField("llm_chat_path", "对话接口路径")}
            {textField("llm_models_path", "模型列表路径")}
            <label className="training-field settings-field"><span>API Key</span><input type="password" value={apiKeyDraft} placeholder={config.llm_api_key || "未配置，留空表示不使用或保持不变"} onChange={(event) => { setApiKeyDraft(event.target.value); setClearApiKey(false); }} /><small className="field-hint">当前值只显示掩码；不修改时留空即可。</small></label>
            <label className="training-check settings-key-action"><input type="checkbox" checked={clearApiKey} onChange={(event) => { setClearApiKey(event.target.checked); setApiKeyDraft(""); }} />保存时清除 API Key</label>
          </div>

          <div className="settings-section-title">请求超时</div>
          <div className="training-grid settings-grid">
            {numberField("llm_request_timeout_seconds", "请求超时（秒）")}
            {numberField("llm_connect_timeout_seconds", "连接超时（秒）")}
            {numberField("llm_stream_idle_timeout_seconds", "流式空闲超时（秒）")}
          </div>

          <div className="settings-section-title">实时纠错</div>
          <div className="training-grid settings-grid">
            <label className="training-check settings-key-action"><input type="checkbox" checked={config.llm_correction_enabled} onChange={(event) => update("llm_correction_enabled", event.target.checked)} />启用实时 LLM 纠错</label>
            {numberField("llm_correction_max_tokens", "纠错最大 Token")}
            <label className="training-field settings-field settings-field-wide"><span>纠错系统提示词</span><textarea value={config.llm_correction_system_prompt} onChange={(event) => update("llm_correction_system_prompt", event.target.value)} /></label>
          </div>

          <div className="settings-section-title">会议纠错</div>
          <div className="training-grid settings-grid">
            <label className="training-check settings-key-action"><input type="checkbox" checked={config.meeting_llm_correction_enabled} onChange={(event) => update("meeting_llm_correction_enabled", event.target.checked)} />启用会议 LLM 候选纠错</label>
            <label className="training-field settings-field settings-field-wide"><span>候选纠错映射</span><textarea value={config.meeting_llm_correction_candidates.join("\n")} placeholder="每行一个，例如：术语A->术语B" onChange={(event) => update("meeting_llm_correction_candidates", event.target.value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean))} /><small className="field-hint">默认仍以确定性规则为主；这里只允许确认显式候选替换。</small></label>
          </div>
        </>}
      </section>
    </main>
  );
}
