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
    return (
      <div className="setting-row">
        <div className="setting-row__label">
          {label}
          {hint && <div className="field__hint">{hint}</div>}
        </div>
        <div className="setting-row__control">
          <input className="input input--mono" value={config[key]} onChange={(event) => update(key, event.target.value)} />
        </div>
      </div>
    );
  }

  function numberField(key: "llm_request_timeout_seconds" | "llm_connect_timeout_seconds" | "llm_stream_idle_timeout_seconds" | "llm_correction_max_tokens", label: string, step = "1") {
    if (!config) return null;
    return (
      <div className="setting-row">
        <div className="setting-row__label">{label}</div>
        <div className="setting-row__control">
          <input className="input input--mono" type="number" min="1" step={step} value={config[key]} onChange={(event) => update(key, Number(event.target.value) as LlmConfig[typeof key])} />
        </div>
      </div>
    );
  }

  return (
    <section className="view" aria-label="管理设置">
      <div className="page-head">
        <div className="page-head__text">
          <div className="eyebrow">05 · 管理设置</div>
          <h1>LLM 连接配置</h1>
          <p>填写服务地址与模型名称即可使用，无需手动编辑 .env。保存后立即对实时对话和新的会议任务生效。</p>
        </div>
        <div className="page-head__actions">
          <span className={`badge ${config?.llm_api_key_configured ? "badge--success" : "badge--warning"}`}>
            {config?.llm_api_key_configured ? "已配置" : "待配置"}
          </span>
        </div>
      </div>

      <div className="settings">
        <div className={`alert ${status.includes("正常") ? "alert--success" : "alert--info"}`}>
          <span>◈</span>
          <div className="alert__body">
            <div className="alert__title">配置状态</div>
            <div className="alert__desc">{status}</div>
          </div>
        </div>

        <div className="card">
          <div className="card__head">
            <h3>服务连接</h3>
            <button className="btn btn--ghost btn--sm" type="button" onClick={() => void loadConfig()} disabled={busy}>重载</button>
            <button className="btn btn--secondary btn--sm" type="button" onClick={() => void checkHealth()} disabled={!config || busy}>测试连接</button>
            <button className="btn btn--primary btn--sm" type="button" onClick={() => void saveConfig()} disabled={!config || busy}>保存并应用</button>
          </div>
          {config && (
            <>
              {textField("llm_base_url", "服务地址", "OpenAI 兼容接口，例如 http://127.0.0.1:8000/v1")}
              {textField("llm_model", "模型名称", "必须与 LLM 服务实际提供的模型 ID 一致")}
              {textField("llm_chat_path", "对话接口路径")}
              {textField("llm_models_path", "模型列表路径")}
              <div className="setting-row">
                <div className="setting-row__label">
                  API Key
                  <div className="field__hint">当前值只显示掩码；不修改时留空即可</div>
                </div>
                <div className="setting-row__control">
                  <div className="setting-inline">
                    <input
                      className="input input--mono"
                      type="password"
                      value={apiKeyDraft}
                      placeholder={config.llm_api_key || "未配置，留空表示不使用或保持不变"}
                      onChange={(event) => {
                        setApiKeyDraft(event.target.value);
                        setClearApiKey(false);
                      }}
                    />
                  </div>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={clearApiKey}
                      onChange={(event) => {
                        setClearApiKey(event.target.checked);
                        setApiKeyDraft("");
                      }}
                    />
                    保存时清除 API Key
                  </label>
                </div>
              </div>
            </>
          )}
        </div>

        {config && (
          <div className="card">
            <div className="card__head"><h3>请求超时</h3></div>
            {numberField("llm_request_timeout_seconds", "请求超时（秒）")}
            {numberField("llm_connect_timeout_seconds", "连接超时（秒）")}
            {numberField("llm_stream_idle_timeout_seconds", "流式空闲超时（秒）")}
          </div>
        )}

        {config && (
          <div className="card">
            <div className="card__head"><h3>实时纠错</h3></div>
            <div className="setting-row">
              <div className="setting-row__label">实时 LLM 纠错</div>
              <div className="setting-row__control setting-inline">
                <button
                  className="switch"
                  role="switch"
                  type="button"
                  aria-checked={config.llm_correction_enabled}
                  aria-label="启用实时 LLM 纠错"
                  onClick={() => update("llm_correction_enabled", !config.llm_correction_enabled)}
                />
                <span className="card__hint">对话过程中对 ASR 结果调用 LLM 纠错</span>
              </div>
            </div>
            {numberField("llm_correction_max_tokens", "纠错最大 Token")}
            <div className="setting-row">
              <div className="setting-row__label">纠错系统提示词</div>
              <div className="setting-row__control">
                <textarea className="textarea" value={config.llm_correction_system_prompt} onChange={(event) => update("llm_correction_system_prompt", event.target.value)} />
              </div>
            </div>
          </div>
        )}

        {config && (
          <div className="card">
            <div className="card__head"><h3>会议纠错</h3></div>
            <div className="setting-row">
              <div className="setting-row__label">会议 LLM 候选纠错</div>
              <div className="setting-row__control setting-inline">
                <button
                  className="switch"
                  role="switch"
                  type="button"
                  aria-checked={config.meeting_llm_correction_enabled}
                  aria-label="启用会议 LLM 候选纠错"
                  onClick={() => update("meeting_llm_correction_enabled", !config.meeting_llm_correction_enabled)}
                />
                <span className="card__hint">关闭时只应用确定性纠错，不调用 LLM 清洗</span>
              </div>
            </div>
            <div className="setting-row">
              <div className="setting-row__label">
                候选纠错映射
                <div className="field__hint">每行一个，例如：术语A-&gt;术语B</div>
              </div>
              <div className="setting-row__control">
                <textarea
                  className="textarea"
                  value={config.meeting_llm_correction_candidates.join("\n")}
                  placeholder="每行一个，例如：术语A->术语B"
                  onChange={(event) =>
                    update(
                      "meeting_llm_correction_candidates",
                      event.target.value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean),
                    )
                  }
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
