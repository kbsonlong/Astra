import { FormEvent, useState } from "react";

type AuthStatus = {
  auth_required: boolean;
  authenticated: boolean;
};

type Props = {
  onAuthenticated: (status: AuthStatus) => void;
};

export default function LoginPage({ onAuthenticated }: Props) {
  const [token, setToken] = useState("");
  const [status, setStatus] = useState("请输入管理员令牌");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!token.trim()) {
      setStatus("请输入管理员令牌");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token }),
      });
      const payload = await response.json() as AuthStatus & { detail?: string };
      if (!response.ok || !payload.authenticated) {
        throw new Error(payload.detail ?? "管理员令牌无效");
      }
      setToken("");
      onAuthenticated(payload);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "登录失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell login-shell">
      <section className="login-card" aria-live="polite">
        <span className="eyebrow">ADMINISTRATOR ACCESS</span>
        <h1>登录 Astra</h1>
        <p>此服务受管理员令牌保护。令牌只用于建立当前浏览器会话，不会保存在本地存储中。</p>
        <form onSubmit={submit} className="login-form">
          <label className="field">
            <span>管理员令牌</span>
            <input
              autoFocus
              type="password"
              autoComplete="current-password"
              value={token}
              onChange={(event) => setToken(event.target.value)}
              placeholder="输入 ADMIN_TOKEN"
            />
          </label>
          <button type="submit" disabled={busy}>{busy ? "登录中…" : "登录"}</button>
        </form>
        <small className="connection-note">{status}</small>
      </section>
    </main>
  );
}
