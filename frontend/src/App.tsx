import { useEffect, useRef, useState } from "react";
import UploadPage from "./UploadPage";

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

export default function App() {
  if (location.pathname === "/upload") return <UploadPage />;

  const socket = useRef<WebSocket | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const mediaStream = useRef<MediaStream | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
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
    recorder.current?.stop();
    recorder.current = null;
    mediaStream.current?.getTracks().forEach((track) => track.stop());
    mediaStream.current = null;
    void audioContext.current?.close();
    audioContext.current = null;
    analyser.current = null;
    speechStarted.current = false;
    silenceSince.current = null;
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
    if (stateRef.current === "LISTENING") {
      if (level >= SPEECH_THRESHOLD) {
        if (!speechStarted.current) {
          speechStarted.current = true;
          setTranscript("");
          setAnswer("");
        }
        silenceSince.current = null;
      } else if (speechStarted.current) {
        silenceSince.current ??= now;
        if (now - silenceSince.current >= SILENCE_MS) {
          const connection = socket.current;
          if (connection?.readyState === WebSocket.OPEN) {
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
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      throw new Error("当前浏览器不支持麦克风录音，请使用 HTTPS 或 localhost");
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStream.current = stream;
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const currentAnalyser = context.createAnalyser();
    currentAnalyser.fftSize = 1024;
    source.connect(currentAnalyser);
    audioContext.current = context;
    analyser.current = currentAnalyser;

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    const currentRecorder = new MediaRecorder(stream, { mimeType });
    currentRecorder.ondataavailable = (event) => {
      if (event.data.size > 0 && connection.readyState === WebSocket.OPEN) {
        connection.send(event.data);
      }
    };
    currentRecorder.onerror = () => setConnectionError("浏览器录音失败，请检查麦克风权限");
    recorder.current = currentRecorder;
    currentRecorder.start(250);
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
    if (event.type === "tts_chunk" && event.audio_b64) {
      const bytes = Uint8Array.from(atob(event.audio_b64), (char) => char.charCodeAt(0));
      new Audio(URL.createObjectURL(new Blob([bytes], { type: event.mime ?? "audio/wav" }))).play();
    }
    if (event.type === "tts_end") {
      stateRef.current = "LISTENING";
      setState("LISTENING");
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
          <a className="nav-link" href="/upload">音频工作台</a>
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
