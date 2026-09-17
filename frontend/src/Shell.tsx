import { ReactNode, useEffect, useState } from "react";

export type NavKey = "live" | "work" | "review" | "train" | "settings";

const NAV_TITLES: Record<NavKey, string> = {
  live: "实时助手",
  work: "音频工作台",
  review: "会议复核",
  train: "模型训练",
  settings: "管理设置",
};

const NAV_HREF: Record<NavKey, string> = {
  live: "/",
  work: "/upload",
  review: "/review",
  train: "/training",
  settings: "/settings",
};

type HealthState = {
  llm: "ok" | "warn" | "off";
  asr: "ok" | "warn" | "off";
  tts: "ok" | "warn" | "off";
  vad: "ok" | "warn" | "off";
};

const HEALTH_LABEL: Record<"ok" | "warn" | "off", string> = {
  ok: "已连接",
  warn: "加载中",
  off: "未就绪",
};

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    if (typeof document !== "undefined") {
      return document.documentElement.getAttribute("data-theme") || "light";
    }
    return "light";
  });
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("astra-ds-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

type Props = {
  active: NavKey;
  title?: string;
  pendingReview?: number;
  onLogout?: () => void;
  children: ReactNode;
};

export default function Shell({ active, title, pendingReview = 0, onLogout, children }: Props) {
  const [theme, toggleTheme] = useTheme();
  const [health, setHealth] = useState<HealthState>({ llm: "off", asr: "off", tts: "off", vad: "off" });

  useEffect(() => {
    let cancelled = false;
    async function probe() {
      try {
        const response = await fetch("/api/health", { credentials: "same-origin" });
        if (!response.ok) return;
        const data = (await response.json()) as {
          llm?: { ok?: boolean };
          asr?: { ok?: boolean };
          tts?: { ok?: boolean };
          meeting?: { workflow?: { vad?: { ok?: boolean } } };
        };
        if (cancelled) return;
        const flag = (ok?: boolean): "ok" | "warn" | "off" => (ok ? "ok" : "warn");
        setHealth({
          llm: flag(data.llm?.ok),
          asr: flag(data.asr?.ok),
          tts: flag(data.tts?.ok),
          vad: flag(data.meeting?.workflow?.vad?.ok),
        });
      } catch {
        /* keep previous */
      }
    }
    void probe();
    const timer = window.setInterval(() => void probe(), 15000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const navGroups: { label: string; items: NavKey[] }[] = [
    { label: "工作区", items: ["live", "work", "review", "train"] },
    { label: "系统", items: ["settings"] },
  ];

  return (
    <div className="app">
      <aside className="rail">
        <div className="rail__brand">
          <span className="rail__mark">Astra</span>
          <span className="rail__sub">Local Meeting Studio</span>
        </div>

        {navGroups.map((group) => (
          <nav className="rail__group" aria-label={group.label} key={group.label}>
            <span className="rail__label">{group.label}</span>
            {group.items.map((key) => (
              <a
                className="navitem"
                href={NAV_HREF[key]}
                key={key}
                aria-current={active === key ? "page" : undefined}
              >
                <span className="navitem__icon">{navIcon(key)}</span>
                {NAV_TITLES[key]}
                {key === "review" && pendingReview > 0 && (
                  <span className="navitem__count">{pendingReview}</span>
                )}
              </a>
            ))}
          </nav>
        ))}

        <div className="health" style={{ marginTop: "auto" }}>
          <div className="health__title">服务状态</div>
          {(["llm", "asr", "tts", "vad"] as const).map((key) => (
            <div className="health__row" key={key}>
              <span className={`health__dot dot--${health[key]}`} />
              <span className="health__name">{key.toUpperCase()}</span>
              <span className="health__val">{HEALTH_LABEL[health[key]]}</span>
            </div>
          ))}
          <div
            className="health__row"
            style={{ marginTop: 6, paddingTop: 8, borderTop: "1px solid var(--border-subtle)" }}
          >
            <span className="health__name">全部数据存于本机</span>
          </div>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <span className="crumb">
            Astra / <strong>{title ?? NAV_TITLES[active]}</strong>
          </span>
          <div className="topbar__spacer" />
          {pendingReview > 0 && (
            <span className="badge badge--warning">
              <span className="badge__dot" />
              {pendingReview} 个声纹待审核
            </span>
          )}
          <button
            className="btn btn--secondary btn--sm"
            type="button"
            onClick={toggleTheme}
            aria-label="切换深浅主题"
          >
            <span>{theme === "dark" ? "◑" : "◐"}</span>
            <span>{theme === "dark" ? "浅色" : "深色"}</span>
          </button>
          {onLogout && (
            <button className="btn btn--ghost btn--sm" type="button" onClick={onLogout}>
              退出登录
            </button>
          )}
        </header>
        <div className="body">{children}</div>
      </div>
    </div>
  );
}

function navIcon(key: NavKey): string {
  switch (key) {
    case "live":
      return "◎";
    case "work":
      return "▤";
    case "review":
      return "≡";
    case "train":
      return "◈";
    case "settings":
      return "⚙";
  }
}
