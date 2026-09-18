const DIAL_BARS = [
  { rot: 0, h: 14, op: 0.55 }, { rot: 22.5, h: 26, op: 0.85 },
  { rot: 45, h: 18, op: 0.6 }, { rot: 67.5, h: 32, op: 1 },
  { rot: 90, h: 20, op: 0.7 }, { rot: 112.5, h: 12, op: 0.45 },
  { rot: 135, h: 28, op: 0.9 }, { rot: 157.5, h: 16, op: 0.6 },
  { rot: 180, h: 22, op: 0.75 }, { rot: 202.5, h: 14, op: 0.5 },
  { rot: 225, h: 30, op: 0.95 }, { rot: 247.5, h: 18, op: 0.65 },
  { rot: 270, h: 24, op: 0.8 }, { rot: 292.5, h: 12, op: 0.45 },
  { rot: 315, h: 20, op: 0.7 }, { rot: 337.5, h: 26, op: 0.85 },
];

export default function VoiceDial({ state, label }: { state: string; label: string }) {
  const active = state === "LISTENING" || state === "SPEAKING";
  const thinking = state === "REASONING";
  const barColor =
    state === "SPEAKING" ? "var(--accent)" : active ? "var(--live)" : "var(--border-strong)";
  const centerFill = active ? "var(--live-soft)" : "var(--bg-inset)";
  const centerStroke =
    state === "SPEAKING" ? "var(--accent)" : active ? "var(--live)" : "var(--border-strong)";
  const centerText =
    state === "SPEAKING" ? "var(--accent-text)" : active ? "var(--live-text)" : "var(--text-tertiary)";

  return (
    <svg width="232" height="232" viewBox="0 0 232 232" role="img" aria-label={`声纹罗盘，当前状态：${label}`}>
      <circle cx="116" cy="116" r="112" fill="none" stroke="var(--border-subtle)" strokeWidth="1" />
      <g className={thinking ? "ring--thinking" : undefined} stroke="var(--border-default)" strokeWidth="1">
        <line x1="116" y1="6" x2="116" y2="14" /><line x1="116" y1="218" x2="116" y2="226" />
        <line x1="6" y1="116" x2="14" y2="116" /><line x1="218" y1="116" x2="226" y2="116" />
      </g>
      <circle cx="116" cy="116" r="84" fill="none" stroke={active ? "var(--live-border)" : "var(--border-subtle)"} strokeWidth="1" strokeDasharray="2 6" />
      <g className={active ? "wv wv--live" : "wv"} fill={barColor}>
        {DIAL_BARS.map((bar) => <g transform={`rotate(${bar.rot} 116 116)`} key={bar.rot}><rect x="113.5" y={68 - bar.h} width="5" height={bar.h} rx="2.5" opacity={bar.op} /></g>)}
      </g>
      <circle cx="116" cy="116" r="40" fill={centerFill} stroke={centerStroke} strokeWidth="1.5" />
      <text x="116" y="121" textAnchor="middle" fontFamily="var(--font-mono)" fontSize="12" fontWeight="600" fill={centerText}>{label}</text>
    </svg>
  );
}
