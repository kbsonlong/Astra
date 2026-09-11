import { useEffect, useRef, useState } from "react";
import UploadPage from "./UploadPage";
import ReviewPage from "./ReviewPage";
import TrainingPage from "./TrainingPage";
import SettingsPage from "./SettingsPage";
import LoginPage from "./LoginPage";
import {
  JAEC_SAMPLE_RATE,
  PCM_FRAME_SAMPLES,
  PcmFrameBuffer,
  encodePcm16Frame,
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
    return () => { active = false; };
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
    return () => { window.fetch = originalFetch; };
  }, []);

  async function logout() {
    await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "same-origin",
    });
    setAuth({ auth_required: true, authenticated: false });
  }

  if (!auth) {
    return <main className="app-shell login-shell"><p>{authError || "正在确认管理员会话…"}</p></main>;
  }
  if (auth.auth_required && !auth.authenticated) {
    return <LoginPage onAuthenticated={setAuth} />;
  }

  const page = location.pathname === "/upload" ? <UploadPage />
    : location.pathname === "/review" ? <ReviewPage />
      : location.pathname === "/training" ? <TrainingPage />
        : location.pathname === "/settings" ? <SettingsPage />
          : <VoiceAssistant />;

  return <>{auth.auth_required && <button className="auth-logout" type="button" onClick={() => void logout()}>退出登录</button>}{page}</>;
}

function VoiceAssistant() {
  if (location.pathname === "/upload") return <UploadPage />;
  if (location.pathname === "/review") return <ReviewPage />;
  if (location.pathname === "/training") return <TrainingPage />;
  if (location.pathname === "/settings") return <SettingsPage />;

  const socket = useRef<WebSocket | null>(null);
  const mediaStream = useRef<MediaStream | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const pcmCapture = useRef<ScriptProcessorNode | null>(null);
  const farEndBus = useRef<GainNode | null>(null);
  const ttsSources = useRef(new Set<AudioBufferSourceNode>());
  const microphoneFrames = useRef(new PcmFrameBuffer());
  const farEndFrames = useRef(new PcmFrameBuffer());
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

  function resetCaptureWindow() {
    microphoneFrames.current.clear();
    farEndFrames.current.clear();
    captureWindow.current = true;
  }

  function stopTtsPlayback() {
    for (const source of ttsSources.current) {
      try { source.stop(); } catch { /* source may already have ended */ }
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
              connection.send(JSON.stringify({
                type: "interrupt",
                generation_id: activeGeneration.current || undefined,
                reason: "vad",
              }));
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
    connection.send(JSON.stringify({
      type: "audio_format",
      format: "pcm16",
      sample_rate: JAEC_SAMPLE_RATE,
      frame_samples: PCM_FRAME_SAMPLES,
    }));
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
      const reference = event.inputBuffer.numberOfChannels > 1
        ? event.inputBuffer.getChannelData(1)
        : new Float32Array(microphone.length);
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
      const frameCount = Math.min(microphoneChunks.length, referenceChunks.length);
      for (let index = 0; index < frameCount; index += 1) {
        const sequence = pcmSequence.current;
        pcmSequence.current += 1;
        if (captureWindow.current && stateRef.current === "LISTENING") {
          connection.send(encodePcm16Frame(
            microphoneChunks[index], sequence, "microphone", JAEC_SAMPLE_RATE,
          ));
        }
        if (captureWindow.current || stateRef.current === "SPEAKING") {
          connection.send(encodePcm16Frame(
            referenceChunks[index], sequence, "reference", JAEC_SAMPLE_RATE,
          ));
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
    socket.current?.send(JSON.stringify({
      type: "interrupt",
      generation_id: activeGeneration.current || undefined,
      reason: "manual",
    }));
    socket.current?.send(JSON.stringify({ type: "end_session" }));
    socket.current?.close();
    cleanupAudio();
    setConnected(false);
    stateRef.current = "IDLE";
    setState("IDLE");
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">A</span>
          <span>
            <strong>Astra</strong>
            <small>Local Voice Assistant</small>
          </span>
        </a>
        <nav className="nav-actions" aria-label="Astra tools">
          <a className="nav-link active" href="/">实时通话</a>
          <a className="nav-link" href="/upload">会议工作台</a>
          <a className="nav-link" href="/review">逐段审校</a>
          <a className="nav-link" href="/training">训练设置</a>
          <a className="nav-link" href="/settings">管理设置</a>
        </nav>
      </header>

      <section className="hero-panel" aria-live="polite">
        <div className="hero-copy">
          <span className="eyebrow">ON-DEVICE SPEECH LOOP</span>
          <h1>更安静、更直接的现代语音助手</h1>
          <p>{stateHints[state] ?? "保持会话连接，Astra 会在回答后回到聆听状态。"}</p>
          <div className="hero-actions">
            {!connected ? (
              <button className="primary-action" onClick={connect}>开始通话</button>
            ) : (
              <button className="primary-action stop" onClick={stop}>停止通话</button>
            )}
            <span className={`status-pill ${connected ? "online" : "offline"}`}>
              <span />
              {stateLabels[state] ?? state}
            </span>
          </div>
          <small className="connection-note">{connectionError || (connected ? "麦克风准备中" : "尚未连接本地服务")}</small>
        </div>

        <div className={`voice-console state-${state.toLowerCase()}`}>
          <div className="assistant-core" aria-hidden="true">
            <span className="core-ring ring-one" />
            <span className="core-ring ring-two" />
            <span className="core-letter">A</span>
          </div>
          <div className="waveform" aria-hidden="true">
            {Array.from({ length: 18 }, (_, index) => <span key={index} />)}
          </div>
          <div className="console-meta">
            <span>Session</span>
            <strong>{connected ? "Live" : "Standby"}</strong>
          </div>
        </div>
      </section>

      <section className="conversation-grid" aria-live="polite">
        <article className="dialog-card user">
          <div className="message-head">
            <span>你</span>
            <small>Speech input</small>
          </div>
          <p>{transcript || "等待语音输入。开始通话后，说完停顿一下即可自动提交。"}</p>
        </article>
        <article className="dialog-card assistant">
          <div className="message-head">
            <span>Astra</span>
            <small>Assistant response</small>
          </div>
          <p>{answer || "准备好后开始对话。我会把回答转成语音播放，并继续等待下一轮输入。"}</p>
        </article>
      </section>
    </main>
  );
}
