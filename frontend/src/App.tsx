import { useEffect, useRef, useState } from "react";
import UploadPage from "./UploadPage";
import ReviewPage from "./ReviewPage";
import TrainingPage from "./TrainingPage";
import SettingsPage from "./SettingsPage";
import LoginPage from "./LoginPage";
import Shell from "./Shell";
import {
  JAEC_SAMPLE_RATE,
  PCM_FRAME_SAMPLES,
  PcmCollector,
  PcmFrameBuffer,
  encodePcm16Frame,
  encodePcm16Wav,
  resamplePcm,
} from "./audio/pcm";

type ServerEvent = {
  type: string;
  state?: string;
  text?: string;
  token?: string;
  generation_id?: number;
  audio_b64?: string;
  mime?: string;
};

const SILENCE_MS = 800;
const SPEECH_THRESHOLD = 0.025;

const socketUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

const stateLabels: Record<string, string> = {
  IDLE: "待机",
  CONNECTING: "连接中",
  LISTENING: "聆听中",
  REASONING: "思考中",
  SPEAKING: "播报中",
};

const stateHints: Record<string, string> = {
  IDLE: "连接本地服务后即可开始语音对话。",
  CONNECTING: "正在建立 WebSocket 与麦克风通道。",
  LISTENING: "可以直接说话，停顿后会自动提交。",
  REASONING: "Astra 正在理解问题并生成回答。",
  SPEAKING: "正在播放语音回答，结束后会继续聆听。",
};

// 罗盘中心状态标签的徽章语义
const stateBadgeClass: Record<string, string> = {
  IDLE: "badge--neutral",
  CONNECTING: "badge--warning",
  LISTENING: "badge--live",
  REASONING: "badge--accent",
  SPEAKING: "badge--accent",
};

type AuthStatus = {
  auth_required: boolean;
  authenticated: boolean;
};

export default function App() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [authError, setAuthError] = useState("");

  useEffect(() => {
    let active = true;
    void fetch("/api/auth/status", { credentials: "same-origin" })
      .then(async (response) => {
        if (!response.ok) throw new Error("无法读取认证状态");
        return response.json() as Promise<AuthStatus>;
      })
      .then((status) => {
        if (active) setAuth(status);
      })
      .catch((error: unknown) => {
        if (active) {
          setAuthError(error instanceof Error ? error.message : "无法读取认证状态");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const response = await originalFetch.call(window, ...args);
      if (response.status === 401) {
        setAuth({ auth_required: true, authenticated: false });
      }
      return response;
    };
    return () => {
      window.fetch = originalFetch;
    };
  }, []);

  async function logout() {
    await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "same-origin",
    });
    setAuth({ auth_required: true, authenticated: false });
  }

  if (!auth) {
    return (
      <main className="login-shell">
        <p>{authError || "正在确认管理员会话…"}</p>
      </main>
    );
  }
  if (auth.auth_required && !auth.authenticated) {
    return <LoginPage onAuthenticated={setAuth} />;
  }

  const onLogout = auth.auth_required ? () => void logout() : undefined;
  const path = location.pathname;

  if (path === "/upload") {
    return (
      <Shell active="work" onLogout={onLogout}>
        <UploadPage />
      </Shell>
    );
  }
  if (path === "/review") {
    return (
      <Shell active="review" onLogout={onLogout}>
        <ReviewPage />
      </Shell>
    );
  }
  if (path === "/training") {
    return (
      <Shell active="train" onLogout={onLogout}>
        <TrainingPage />
      </Shell>
    );
  }
  if (path === "/settings") {
    return (
      <Shell active="settings" onLogout={onLogout}>
        <SettingsPage />
      </Shell>
    );
  }
  return (
    <Shell active="live" onLogout={onLogout}>
      <VoiceAssistant />
    </Shell>
  );
}

function VoiceAssistant() {
  const socket = useRef<WebSocket | null>(null);
  const mediaStream = useRef<MediaStream | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const pcmCapture = useRef<ScriptProcessorNode | null>(null);
  const farEndBus = useRef<GainNode | null>(null);
  const ttsSources = useRef(new Set<AudioBufferSourceNode>());
  const microphoneFrames = useRef(new PcmFrameBuffer());
  const farEndFrames = useRef(new PcmFrameBuffer());
  const recordingPcm = useRef(new PcmCollector());
  const recordingSampleRate = useRef(JAEC_SAMPLE_RATE);
  const pcmSequence = useRef(0);
  const captureWindow = useRef(false);
  const nextTtsTime = useRef(0);
  const monitorFrame = useRef<number | null>(null);
  const speechStarted = useRef(false);
  const silenceSince = useRef<number | null>(null);
  const stateRef = useRef("IDLE");
  const activeGeneration = useRef(0);
  const [state, setState] = useState("IDLE");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [connected, setConnected] = useState(false);
  const [recordingReady, setRecordingReady] = useState(false);
  const [connectionError, setConnectionError] = useState("");

  useEffect(() => () => cleanupAudio(), []);

  function cleanupAudio() {
    if (monitorFrame.current !== null) cancelAnimationFrame(monitorFrame.current);
    monitorFrame.current = null;
    pcmCapture.current?.disconnect();
    pcmCapture.current = null;
    farEndBus.current?.disconnect();
    farEndBus.current = null;
    stopTtsPlayback();
    microphoneFrames.current.clear();
    farEndFrames.current.clear();
    pcmSequence.current = 0;
    captureWindow.current = false;
    nextTtsTime.current = 0;
    mediaStream.current?.getTracks().forEach((track) => track.stop());
    mediaStream.current = null;
    void audioContext.current?.close();
    audioContext.current = null;
    analyser.current = null;
    speechStarted.current = false;
    silenceSince.current = null;
  }

  function clearLocalRecording() {
    recordingPcm.current.clear();
    setRecordingReady(false);
  }

  function saveLocalRecording() {
    const samples = recordingPcm.current.toSampleRate(
      recordingSampleRate.current,
      JAEC_SAMPLE_RATE,
    );
    if (samples.length === 0) {
      setConnectionError("当前通话没有采集到可保存的录音");
      return;
    }
    const wav = encodePcm16Wav(samples, JAEC_SAMPLE_RATE);
    const url = URL.createObjectURL(new Blob([wav], { type: "audio/wav" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `astra-recording-${new Date().toISOString().replace(/[:.]/g, "-")}.wav`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    setConnectionError("录音已保存到本地下载目录");
  }

  function resetCaptureWindow() {
    microphoneFrames.current.clear();
    farEndFrames.current.clear();
    captureWindow.current = true;
  }

  function stopTtsPlayback() {
    for (const source of ttsSources.current) {
      try {
        source.stop();
      } catch {
        /* source may already have ended */
      }
      source.disconnect();
    }
    ttsSources.current.clear();
    nextTtsTime.current = 0;
  }

  function monitorMicrophone() {
    const currentAnalyser = analyser.current;
    if (!currentAnalyser) return;
    const samples = new Uint8Array(currentAnalyser.fftSize);
    currentAnalyser.getByteTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) {
      const value = (sample - 128) / 128;
      sum += value * value;
    }
    const level = Math.sqrt(sum / samples.length);
    const now = performance.now();
    if (stateRef.current === "LISTENING" || stateRef.current === "SPEAKING") {
      if (level >= SPEECH_THRESHOLD) {
        if (!speechStarted.current) {
          speechStarted.current = true;
          setTranscript("");
          setAnswer("");
          if (stateRef.current === "SPEAKING") {
            const connection = socket.current;
            if (connection?.readyState === WebSocket.OPEN) {
              connection.send(
                JSON.stringify({
                  type: "interrupt",
                  generation_id: activeGeneration.current || undefined,
                  reason: "vad",
                }),
              );
            }
            stopTtsPlayback();
            stateRef.current = "LISTENING";
            setState("LISTENING");
            resetCaptureWindow();
          }
        }
        silenceSince.current = null;
      } else if (speechStarted.current) {
        silenceSince.current ??= now;
        if (now - silenceSince.current >= SILENCE_MS) {
          const connection = socket.current;
          if (connection?.readyState === WebSocket.OPEN) {
            captureWindow.current = false;
            connection.send(JSON.stringify({ type: "speech_end" }));
            stateRef.current = "REASONING";
            setState("REASONING");
          }
          speechStarted.current = false;
          silenceSince.current = null;
        }
      }
    } else {
      speechStarted.current = false;
      silenceSince.current = null;
    }
    monitorFrame.current = requestAnimationFrame(monitorMicrophone);
  }

  async function startMicrophone(connection: WebSocket) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前浏览器不支持麦克风录音，请使用 HTTPS 或 localhost");
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStream.current = stream;
    const context = new AudioContext({ sampleRate: JAEC_SAMPLE_RATE });
    recordingSampleRate.current = context.sampleRate;
    connection.send(
      JSON.stringify({
        type: "audio_format",
        format: "pcm16",
        sample_rate: JAEC_SAMPLE_RATE,
        frame_samples: PCM_FRAME_SAMPLES,
      }),
    );
    const source = context.createMediaStreamSource(stream);
    const currentAnalyser = context.createAnalyser();
    currentAnalyser.fftSize = 1024;
    source.connect(currentAnalyser);
    const currentFarEndBus = context.createGain();
    currentFarEndBus.connect(context.destination);
    const merger = context.createChannelMerger(2);
    source.connect(merger, 0, 0);
    currentFarEndBus.connect(merger, 0, 1);
    const currentPcmCapture = context.createScriptProcessor(4096, 2, 2);
    currentPcmCapture.onaudioprocess = (event) => {
      const microphone = event.inputBuffer.getChannelData(0);
      const reference =
        event.inputBuffer.numberOfChannels > 1
          ? event.inputBuffer.getChannelData(1)
          : new Float32Array(microphone.length);
      recordingPcm.current.append(microphone);
      setRecordingReady(true);
      microphoneFrames.current.append(
        context.sampleRate === JAEC_SAMPLE_RATE
          ? microphone
          : resamplePcm(microphone, context.sampleRate, JAEC_SAMPLE_RATE),
      );
      farEndFrames.current.append(
        context.sampleRate === JAEC_SAMPLE_RATE
          ? reference
          : resamplePcm(reference, context.sampleRate, JAEC_SAMPLE_RATE),
      );
      const microphoneChunks = microphoneFrames.current.drain(PCM_FRAME_SAMPLES);
      const referenceChunks = farEndFrames.current.drain(PCM_FRAME_SAMPLES);
      for (let index = 0; index < microphoneChunks.length; index += 1) {
        const sequence = pcmSequence.current;
        pcmSequence.current += 1;
        if (
          connection.readyState === WebSocket.OPEN &&
          captureWindow.current &&
          stateRef.current === "LISTENING"
        ) {
          connection.send(
            encodePcm16Frame(microphoneChunks[index], sequence, "microphone", JAEC_SAMPLE_RATE),
          );
        }
        const reference = referenceChunks[index];
        if (
          reference &&
          connection.readyState === WebSocket.OPEN &&
          (captureWindow.current || stateRef.current === "SPEAKING")
        ) {
          connection.send(encodePcm16Frame(reference, sequence, "reference", JAEC_SAMPLE_RATE));
        }
      }
    };
    merger.connect(currentPcmCapture);
    const silentSink = context.createGain();
    silentSink.gain.value = 0;
    currentPcmCapture.connect(silentSink);
    silentSink.connect(context.destination);
    audioContext.current = context;
    analyser.current = currentAnalyser;
    pcmCapture.current = currentPcmCapture;
    farEndBus.current = currentFarEndBus;
    resetCaptureWindow();
    monitorFrame.current = requestAnimationFrame(monitorMicrophone);
  }

  async function connect() {
    if (socket.current?.readyState === WebSocket.OPEN) return;
    setConnectionError("");
    clearLocalRecording();
    setState("CONNECTING");
    const connection = new WebSocket(socketUrl);
    connection.onopen = () => {
      setConnected(true);
      connection.send(JSON.stringify({ type: "start_session" }));
      void startMicrophone(connection)
        .then(() => setConnectionError("麦克风已连接，说话后自动提交"))
        .catch((error: unknown) => {
          setConnectionError(error instanceof Error ? error.message : "无法访问麦克风");
          connection.close();
        });
    };
    connection.onclose = () => {
      setConnected(false);
      cleanupAudio();
      setState("IDLE");
    };
    connection.onerror = () => setConnectionError("无法连接到后端，请确认服务已启动");
    connection.onmessage = (message) => {
      try {
        handleEvent(JSON.parse(message.data) as ServerEvent);
      } catch {
        setConnectionError("后端返回了无法识别的消息");
      }
    };
    socket.current = connection;
  }

  function handleEvent(event: ServerEvent) {
    if (event.generation_id && event.type !== "state_change") {
      if (event.generation_id < activeGeneration.current) return;
      activeGeneration.current = event.generation_id;
    }
    if (event.type === "state_change" && event.state) {
      stateRef.current = event.state;
      setState(event.state);
    }
    if (event.type === "asr_final") setTranscript(event.text ?? "");
    if (event.type === "llm_token") setAnswer((current) => current + (event.token ?? ""));
    if (event.type === "tts_start") {
      captureWindow.current = false;
    }
    if (event.type === "tts_chunk" && event.audio_b64) {
      void playTtsChunk(event);
    }
    if (event.type === "tts_end") {
      captureWindow.current = true;
      stateRef.current = "LISTENING";
      setState("LISTENING");
    }
  }

  async function playTtsChunk(event: ServerEvent) {
    const context = audioContext.current;
    if (!context || !event.audio_b64) return;
    const bytes = Uint8Array.from(atob(event.audio_b64), (char) => char.charCodeAt(0));
    try {
      const audioBuffer = await context.decodeAudioData(bytes.buffer.slice(0));
      const source = context.createBufferSource();
      source.buffer = audioBuffer;
      const farEnd = farEndBus.current;
      if (farEnd) source.connect(farEnd);
      else source.connect(context.destination);
      const startAt = Math.max(context.currentTime, nextTtsTime.current);
      source.start(startAt);
      nextTtsTime.current = startAt + audioBuffer.duration;
      ttsSources.current.add(source);
      source.onended = () => {
        ttsSources.current.delete(source);
        source.disconnect();
      };
    } catch {
      setConnectionError("无法解码语音回答");
    }
  }

  function stop() {
    socket.current?.send(
      JSON.stringify({
        type: "interrupt",
        generation_id: activeGeneration.current || undefined,
        reason: "manual",
      }),
    );
    socket.current?.send(JSON.stringify({ type: "end_session" }));
    socket.current?.close();
    cleanupAudio();
    setConnected(false);
    stateRef.current = "IDLE";
    setState("IDLE");
  }

  const listening = state === "LISTENING";
  const centerLabel = stateLabels[state] ?? state;

  return (
    <section className="view" aria-label="实时助手">
      <div className="page-head">
        <div className="page-head__text">
          <div className="eyebrow">01 · 实时助手</div>
          <h1>直接说话，停顿后自动提交</h1>
          <p>{stateHints[state] ?? "保持会话连接，Astra 会在回答后回到聆听状态。"}</p>
        </div>
      </div>

      <div className="live-grid">
        {/* 声纹罗盘 */}
        <div className="card dial-card">
          <div className="dial-stage">
            <VoiceDial state={state} label={centerLabel} />
          </div>
          <div style={{ textAlign: "center" }}>
            <div className="dial-state">{centerLabel}</div>
            <p className="dial-sub" style={{ marginTop: "var(--space-2)" }}>
              {stateHints[state] ?? "连接本地服务后即可开始语音对话。"}
            </p>
          </div>
          <div className="dial-controls">
            {!connected ? (
              <button className="btn btn--primary btn--sm" type="button" onClick={connect}>
                开始通话
              </button>
            ) : (
              <button className="btn btn--danger btn--sm" type="button" onClick={stop}>
                停止通话
              </button>
            )}
            <button
              className="btn btn--secondary btn--sm"
              type="button"
              onClick={saveLocalRecording}
              disabled={!recordingReady || connected}
              title={connected ? "停止通话后保存录音" : "保存本次通话的麦克风录音"}
            >
              保存本地录音
            </button>
          </div>
          <div className="dial-meta">
            <div>
              <div className="dial-meta__k">会话状态</div>
              <div className="dial-meta__v">{connected ? "Live" : "Standby"}</div>
            </div>
            <div>
              <div className="dial-meta__k">当前阶段</div>
              <div className="dial-meta__v">{centerLabel}</div>
            </div>
            <div>
              <div className="dial-meta__k">输入设备</div>
              <div className="dial-meta__v">Mac mini 麦克风</div>
            </div>
            <div>
              <div className="dial-meta__k">采样率</div>
              <div className="dial-meta__v">16 kHz</div>
            </div>
          </div>
        </div>

        {/* 实时流 */}
        <div className="col">
          <div className="card">
            <div className="card__head">
              <h4>本次会话</h4>
              <span className={`badge ${stateBadgeClass[state] ?? "badge--neutral"}`}>
                <span className="badge__dot" />
                {connected ? "进行中" : "未连接"}
              </span>
            </div>
            <div className="spine">
              <div className="spine-row">
                <div className="spine-time">你</div>
                <div className="spine-body">
                  <div className="spine-speaker">语音输入</div>
                  <div className={`spine-text${transcript ? "" : " spine-text--dim"}`}>
                    {transcript || "等待语音输入。开始通话后，说完停顿一下即可自动提交。"}
                  </div>
                </div>
              </div>
              <div className={`spine-row${listening ? "" : " spine-row--live"}`}>
                <div className="spine-time">Astra</div>
                <div className="spine-body">
                  <div className="reply">
                    <div className="reply__label">
                      {state === "REASONING" || state === "SPEAKING"
                        ? "Astra · 正在生成"
                        : "Astra · 本地 LLM"}
                    </div>
                    <div className="reply__text">
                      {answer || "准备好后开始对话。我会把回答转成语音播放，并继续等待下一轮输入。"}
                      {(state === "REASONING" || state === "SPEAKING") && (
                        <span className="caret" aria-hidden="true" />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className={`alert ${connectionError ? "alert--info" : "alert--info"}`}>
            <span>◈</span>
            <div className="alert__body">
              <div className="alert__title">{stateLabels[state] ?? state}</div>
              <div className="alert__desc">
                {connectionError || (connected ? "麦克风准备中" : "尚未连接本地服务")}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

const DIAL_BARS = [
  { rot: 0, h: 14, op: 0.55 },
  { rot: 22.5, h: 26, op: 0.85 },
  { rot: 45, h: 18, op: 0.6 },
  { rot: 67.5, h: 32, op: 1 },
  { rot: 90, h: 20, op: 0.7 },
  { rot: 112.5, h: 12, op: 0.45 },
  { rot: 135, h: 28, op: 0.9 },
  { rot: 157.5, h: 16, op: 0.6 },
  { rot: 180, h: 22, op: 0.75 },
  { rot: 202.5, h: 14, op: 0.5 },
  { rot: 225, h: 30, op: 0.95 },
  { rot: 247.5, h: 18, op: 0.65 },
  { rot: 270, h: 24, op: 0.8 },
  { rot: 292.5, h: 12, op: 0.45 },
  { rot: 315, h: 20, op: 0.7 },
  { rot: 337.5, h: 26, op: 0.85 },
];

function VoiceDial({ state, label }: { state: string; label: string }) {
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
    <svg
      width="232"
      height="232"
      viewBox="0 0 232 232"
      role="img"
      aria-label={`声纹罗盘，当前状态：${label}`}
    >
      <circle cx="116" cy="116" r="112" fill="none" stroke="var(--border-subtle)" strokeWidth="1" />
      <g className={thinking ? "ring--thinking" : undefined} stroke="var(--border-default)" strokeWidth="1">
        <line x1="116" y1="6" x2="116" y2="14" />
        <line x1="116" y1="218" x2="116" y2="226" />
        <line x1="6" y1="116" x2="14" y2="116" />
        <line x1="218" y1="116" x2="226" y2="116" />
      </g>
      <circle
        cx="116"
        cy="116"
        r="84"
        fill="none"
        stroke={active ? "var(--live-border)" : "var(--border-subtle)"}
        strokeWidth="1"
        strokeDasharray="2 6"
      />
      <g className={active ? "wv wv--live" : "wv"} fill={barColor}>
        {DIAL_BARS.map((bar) => (
          <g transform={`rotate(${bar.rot} 116 116)`} key={bar.rot}>
            <rect x="113.5" y={68 - bar.h} width="5" height={bar.h} rx="2.5" opacity={bar.op} />
          </g>
        ))}
      </g>
      <circle cx="116" cy="116" r="40" fill={centerFill} stroke={centerStroke} strokeWidth="1.5" />
      <text
        x="116"
        y="121"
        textAnchor="middle"
        fontFamily="var(--font-mono)"
        fontSize="12"
        fontWeight="600"
        fill={centerText}
      >
        {label}
      </text>
    </svg>
  );
}
