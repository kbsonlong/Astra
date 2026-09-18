import { useEffect, useRef, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import UploadPage from "./UploadPage";
import ReviewPage from "./ReviewPage";
import TrainingPage from "./TrainingPage";
import SettingsPage from "./SettingsPage";
import LoginPage from "./LoginPage";
import Shell from "./Shell";
import VoiceDial from "./live/VoiceDial";
import {
  acceptsGenerationEvent,
  generationForNewConnection,
  nextActiveGeneration,
} from "./live/session";
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
  recording_id?: string;
  download_url?: string;
  code?: string;
  max_bytes?: number;
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
  return (
    <Routes>
      <Route path="/" element={<Shell active="live" onLogout={onLogout}><VoiceAssistant /></Shell>} />
      <Route path="/upload" element={<Shell active="work" onLogout={onLogout}><UploadPage /></Shell>} />
      <Route path="/review" element={<Shell active="review" onLogout={onLogout}><ReviewPage /></Shell>} />
      <Route path="/training" element={<Shell active="train" onLogout={onLogout}><TrainingPage /></Shell>} />
      <Route path="/settings" element={<Shell active="settings" onLogout={onLogout}><SettingsPage /></Shell>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
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
  const recordingId = useRef<string | null>(null);
  const pcmSequence = useRef(0);
  const captureWindow = useRef(false);
  const nextTtsTime = useRef(0);
  const monitorFrame = useRef<number | null>(null);
  const speechStarted = useRef(false);
  const silenceSince = useRef<number | null>(null);
  const stateRef = useRef("IDLE");
  const activeGeneration = useRef(0);
  // 断线重连与录音会话保持
  const manualStop = useRef(false);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<number | null>(null);
  const micActive = useRef(false);
  const [state, setState] = useState("IDLE");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [connected, setConnected] = useState(false);
  const [recordingReady, setRecordingReady] = useState(false);
  const [recordingDownloadUrl, setRecordingDownloadUrl] = useState("");
  const [stopping, setStopping] = useState(false);
  const [connectionError, setConnectionError] = useState("");

  useEffect(() => () => {
    // 客户端路由切换不会再触发整页卸载；显式关闭连接，避免离开实时页后继续重连。
    manualStop.current = true;
    socket.current?.close();
    cleanupAudio();
  }, []);

  // 标签页回到前台时,尝试恢复被浏览器挂起的 AudioContext,避免录音静默中断。
  useEffect(() => {
    function onVisible() {
      if (document.visibilityState !== "visible") return;
      const context = audioContext.current;
      if (context && context.state === "suspended" && micActive.current) {
        void context.resume().catch(() => {
          /* ignore */
        });
      }
    }
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, []);

  function cleanupAudio() {
    if (reconnectTimer.current !== null) {
      window.clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    micActive.current = false;
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

  function prepareServerRecording() {
    recordingId.current = crypto.randomUUID();
    setRecordingDownloadUrl("");
    setRecordingReady(false);
  }

  function saveLocalRecording() {
    if (!recordingDownloadUrl) {
      setConnectionError("服务端录音尚未整理完成");
      return;
    }
    const link = document.createElement("a");
    link.href = recordingDownloadUrl;
    link.click();
    setConnectionError("正在从 Mac mini 下载录音");
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
    // 标签页切后台或系统休眠会让 AudioContext 进入 suspended,onaudioprocess 停止,
    // 录音静默中断。监听 statechange 并尝试自动 resume。
    context.onstatechange = () => {
      if (context.state === "suspended" && micActive.current) {
        void context.resume().catch(() => {
          /* resume 可能因缺少用户手势失败;可见性恢复时会再试 */
        });
      }
    };
    micActive.current = true;
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
      // 用实时的 socket.current 而非闭包捕获的连接:重连后旧连接已失效,
      // 新连接由 socket.current 指向,保证录音会话跨重连持续送帧。
      for (let index = 0; index < microphoneChunks.length; index += 1) {
        const sequence = pcmSequence.current;
        pcmSequence.current += 1;
        const live = socket.current;
        // 所有麦克风帧均上传：后端只把聆听窗口的帧交给 ASR，其余帧不进入
        // 会话缓冲，但会持续追加到服务端录音，避免思考/播报期间留下空洞。
        if (live?.readyState === WebSocket.OPEN) {
          live.send(
            encodePcm16Frame(microphoneChunks[index], sequence, "microphone", JAEC_SAMPLE_RATE),
          );
        }
        const reference = referenceChunks[index];
        if (
          reference &&
          live?.readyState === WebSocket.OPEN &&
          (captureWindow.current || stateRef.current === "SPEAKING")
        ) {
          live.send(encodePcm16Frame(reference, sequence, "reference", JAEC_SAMPLE_RATE));
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
    manualStop.current = false;
    reconnectAttempts.current = 0;
    setConnectionError("");
    prepareServerRecording();
    setStopping(false);
    setState("CONNECTING");
    openSocket(true);
  }

  function openSocket(startMic: boolean) {
    if (reconnectTimer.current !== null) {
      window.clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    const connection = new WebSocket(socketUrl);
    connection.onopen = () => {
      // 服务端为每个 WebSocket 创建一个新的 Session，generation 会从 0 开始。
      // 因此重连时必须丢弃旧连接的代际水位，不能把新事件当作过期事件过滤。
      activeGeneration.current = generationForNewConnection();
      setConnected(true);
      reconnectAttempts.current = 0;
      connection.send(
        JSON.stringify({ type: "start_session", recording_id: recordingId.current || undefined }),
      );
      if (startMic && !micActive.current) {
        void startMicrophone(connection)
          .then(() => setConnectionError("麦克风已连接，说话后自动提交"))
          .catch((error: unknown) => {
            setConnectionError(error instanceof Error ? error.message : "无法访问麦克风");
            manualStop.current = true;
            connection.close();
          });
      } else if (micActive.current) {
        // 重连成功:麦克风与录音会话仍在,只需重发握手并回到聆听。
        connection.send(
          JSON.stringify({
            type: "audio_format",
            format: "pcm16",
            sample_rate: JAEC_SAMPLE_RATE,
            frame_samples: PCM_FRAME_SAMPLES,
          }),
        );
        captureWindow.current = true;
        stateRef.current = "LISTENING";
        setState("LISTENING");
        setConnectionError("连接已恢复，继续聆听");
      }
    };
    connection.onclose = () => {
      setConnected(false);
      if (manualStop.current) {
        cleanupAudio();
        setStopping(false);
        setState("IDLE");
        return;
      }
      // 意外断连:保留麦克风与录音,自动重连(指数退避,最长 10s)。
      scheduleReconnect();
    };
    connection.onerror = () => {
      if (!manualStop.current) {
        setConnectionError("连接中断，正在尝试重连…");
      }
    };
    connection.onmessage = (message) => {
      try {
        handleEvent(JSON.parse(message.data) as ServerEvent);
      } catch {
        setConnectionError("后端返回了无法识别的消息");
      }
    };
    socket.current = connection;
  }

  function scheduleReconnect() {
    if (manualStop.current) return;
    if (reconnectTimer.current !== null) return;
    const attempt = reconnectAttempts.current;
    reconnectAttempts.current = attempt + 1;
    const delay = Math.min(10000, 500 * 2 ** Math.min(attempt, 5));
    stateRef.current = "CONNECTING";
    setState("CONNECTING");
    setConnectionError(`连接已断开，${Math.round(delay / 1000)}s 后自动重连（第 ${attempt + 1} 次）`);
    reconnectTimer.current = window.setTimeout(() => {
      reconnectTimer.current = null;
      if (manualStop.current) return;
      // 麦克风仍在采集,只重开 socket。
      openSocket(!micActive.current);
    }, delay);
  }

  function handleEvent(event: ServerEvent) {
    if (!acceptsGenerationEvent(event, activeGeneration.current)) return;
    activeGeneration.current = nextActiveGeneration(event, activeGeneration.current);
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
    if (event.type === "recording_started" && event.recording_id) {
      recordingId.current = event.recording_id;
    }
    if (event.type === "recording_ready") {
      if (event.download_url) {
        setRecordingDownloadUrl(event.download_url);
        setRecordingReady(true);
        setConnectionError("录音已整理完成，可下载到本地");
      } else {
        setConnectionError("本次通话没有可导出的音频");
      }
      socket.current?.close();
    }
    if (event.type === "error" && event.code === "recording_too_large") {
      const maxMiB = Math.round((event.max_bytes ?? 0) / 1024 / 1024);
      setConnectionError(`服务端录音达到 ${maxMiB} MiB 上限；实时通话继续进行`);
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
    if (stopping) return;
    manualStop.current = true;
    setStopping(true);
    if (reconnectTimer.current !== null) {
      window.clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (socket.current?.readyState === WebSocket.OPEN) {
      socket.current.send(
        JSON.stringify({
          type: "interrupt",
          generation_id: activeGeneration.current || undefined,
          reason: "manual",
        }),
      );
      socket.current.send(JSON.stringify({ type: "end_session" }));
      cleanupAudio();
      setConnectionError("正在将服务端 PCM 录音整理为 WAV…");
      return;
    }
    socket.current?.close();
    cleanupAudio();
    setConnected(false);
    setStopping(false);
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
              <button className="btn btn--danger btn--sm" type="button" onClick={stop} disabled={stopping}>
                {stopping ? "正在结束…" : "停止通话"}
              </button>
            )}
            <button
              className="btn btn--secondary btn--sm"
              type="button"
              onClick={saveLocalRecording}
              disabled={!recordingReady || connected}
              title={connected ? "停止通话后导出服务端录音" : "下载服务端保存的本次通话录音"}
            >
              下载录音
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
