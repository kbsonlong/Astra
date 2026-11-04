import { useEffect, useRef, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import UploadPage from "./UploadPage";
import ReviewPage from "./ReviewPage";
import TrainingPage from "./TrainingPage";
import SettingsPage from "./SettingsPage";
import LoginPage from "./LoginPage";
import Shell from "./Shell";
import VoiceDial from "./live/VoiceDial";
import VoiceLibraryPage from "./tts/VoiceLibraryPage";
import { getTtsHealth, getVoice, listVoices } from "./tts/voiceApi";
import { sessionSelectableVoices } from "./tts/voiceState";
import type { TtsHealth, TtsVoice } from "./tts/voice";
import {
  acceptsGenerationEvent,
  formatElapsedTime,
  generationForNewConnection,
  HEARTBEAT_INTERVAL_MS,
  heartbeatExpired,
  nextActiveGeneration,
  ReconnectPcmBuffer,
  startSessionPayload,
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
  message?: string;
  max_bytes?: number;
};

const SILENCE_MS = 800;
const SPEECH_THRESHOLD = 0.025;
const MAX_SOCKET_BUFFERED_BYTES = 2 * 1024 * 1024;
const PCM_FRAME_DURATION_MS = PCM_FRAME_SAMPLES / JAEC_SAMPLE_RATE * 1000;
const MAX_TRANSCRIPT_CHARACTERS = 50_000;

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

type SessionVoice = { voiceId: string; revision: number; label: string; backend: string };
type MicAccessIssue = "permission" | "device" | "unavailable";
type MicDevice = { deviceId: string; label: string };

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
      <Route path="/voices" element={<Shell active="voices" onLogout={onLogout}><VoiceLibraryPage /></Shell>} />
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
  const reconnectFrames = useRef(new ReconnectPcmBuffer(PCM_FRAME_DURATION_MS));
  const recordingId = useRef<string | null>(null);
  const recordingConfirmed = useRef(false);
  const sessionActiveRef = useRef(false);
  const finalizeAfterReconnect = useRef(false);
  const pcmSequence = useRef(0);
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
  const heartbeatTimer = useRef<number | null>(null);
  const elapsedTimer = useRef<number | null>(null);
  const lastServerMessageAt = useRef(0);
  const sessionStartedAt = useRef<number | null>(null);
  const captureSuspendedAt = useRef<number | null>(null);
  const droppedAudioSecondsRef = useRef(0);
  const pendingSpeechEnd = useRef(false);
  const micActive = useRef(false);
  const [state, setState] = useState("IDLE");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [connected, setConnected] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [recordingStarted, setRecordingStarted] = useState(false);
  const [recordingReady, setRecordingReady] = useState(false);
  const [recordingDownloadUrl, setRecordingDownloadUrl] = useState("");
  const [stopping, setStopping] = useState(false);
  const [connectionError, setConnectionError] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [reconnectCount, setReconnectCount] = useState(0);
  const [droppedAudioSeconds, setDroppedAudioSeconds] = useState(0);
  const [micAccessIssue, setMicAccessIssue] = useState<MicAccessIssue | null>(null);
  const [microphoneDevices, setMicrophoneDevices] = useState<MicDevice[]>([]);
  const [selectedMicDeviceId, setSelectedMicDeviceId] = useState("");
  const [voiceChoices, setVoiceChoices] = useState<TtsVoice[]>([]);
  const [ttsHealth, setTtsHealth] = useState<TtsHealth>();
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const sessionVoice = useRef<SessionVoice | null>(null);

  async function refreshMicrophoneDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const microphones = devices
        .filter((device) => device.kind === "audioinput")
        .map((device, index) => ({
          deviceId: device.deviceId,
          label: device.label || `麦克风 ${index + 1}`,
        }));
      setMicrophoneDevices(microphones);
      setSelectedMicDeviceId((current) => microphones.some((device) => device.deviceId === current) ? current : "");
    } catch {
      setMicrophoneDevices([]);
    }
  }

  useEffect(() => {
    void refreshMicrophoneDevices();
    const onDeviceChange = () => void refreshMicrophoneDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", onDeviceChange);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", onDeviceChange);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void Promise.all([listVoices({ status: "active", pageSize: 200 }, { signal: controller.signal }), getTtsHealth({ signal: controller.signal })])
      .then(([result, health]) => {
        const choices = sessionSelectableVoices(result.items, health);
        setVoiceChoices(choices);
        setTtsHealth(health);
        const requestedVoice = new URLSearchParams(location.search).get("voice_id");
        const requestedRevision = Number(new URLSearchParams(location.search).get("voice_revision"));
        if (requestedVoice && choices.some((voice) => voice.voice_id === requestedVoice && voice.revision?.revision === requestedRevision)) {
          setSelectedVoiceId(requestedVoice);
          setConnectionError("已预选来自音色库的回复音色；开始通话后将固定该版本。");
        }
      })
      .catch(() => setConnectionError("无法读取可用音色；仍可使用系统默认 TTS。"));
    return () => controller.abort();
  }, []);

  const selectedVoice = voiceChoices.find((voice) => voice.voice_id === selectedVoiceId);

  useEffect(() => {
    if (!connected || !sessionVoice.current?.voiceId) return;
    let active = true;
    const checkPinnedVoice = () => {
      const voiceId = sessionVoice.current?.voiceId;
      if (!voiceId) return;
      void Promise.all([getVoice(voiceId), getTtsHealth()]).then(([voice, health]) => {
        if (!active || sessionSelectableVoices([voice], health).length) return;
        setConnectionError("当前会话的回复音色已撤销授权或不可用。请结束通话；新的会话不会继续使用该音色。");
      }).catch(() => {
        if (active) setConnectionError("无法确认当前音色状态；如需继续，请改用系统默认 TTS 后重新开始通话。");
      });
    };
    checkPinnedVoice();
    const timer = window.setInterval(checkPinnedVoice, 15000);
    return () => { active = false; window.clearInterval(timer); };
  }, [connected]);

  function selectedSessionVoice(): SessionVoice | null {
    const revision = selectedVoice?.revision;
    if (!selectedVoice || !revision) return null;
    return {
      voiceId: selectedVoice.voice_id,
      revision: revision.revision,
      label: selectedVoice.display_name,
      backend: revision.backend_family,
    };
  }

  useEffect(() => () => {
    // 客户端路由切换不会再触发整页卸载；显式关闭连接，避免离开实时页后继续重连。
    manualStop.current = true;
    socket.current?.close();
    cleanupAudio();
    stopElapsedClock();
  }, []);

  // 标签页回到前台时,尝试恢复被浏览器挂起的 AudioContext,避免录音静默中断。
  useEffect(() => {
    function onVisible() {
      if (document.visibilityState !== "visible") return;
      const context = audioContext.current;
      if (context && context.state === "suspended" && micActive.current) {
        void context.resume().catch(() => {
          setConnectionError("浏览器暂停了麦克风；请保持页面前台并点击页面后重试");
        });
      }
    }
    document.addEventListener("visibilitychange", onVisible);
    return () => document.removeEventListener("visibilitychange", onVisible);
  }, []);

  useEffect(() => {
    function protectActiveRecording(event: MouseEvent) {
      if (!sessionActiveRef.current || !(event.target instanceof Element)) return;
      const anchor = event.target.closest("a[href]");
      if (!(anchor instanceof HTMLAnchorElement)) return;
      const destination = new URL(anchor.href, location.href);
      if (destination.origin !== location.origin || destination.pathname === location.pathname) return;
      event.preventDefault();
      setConnectionError("录音仍在进行，请先点击“停止并保存”再离开实时页面");
    }

    function protectPageClose(event: BeforeUnloadEvent) {
      if (!sessionActiveRef.current) return;
      event.preventDefault();
      event.returnValue = "";
    }

    document.addEventListener("click", protectActiveRecording, true);
    window.addEventListener("beforeunload", protectPageClose);
    return () => {
      document.removeEventListener("click", protectActiveRecording, true);
      window.removeEventListener("beforeunload", protectPageClose);
    };
  }, []);

  function setSessionIsActive(active: boolean) {
    sessionActiveRef.current = active;
    setSessionActive(active);
  }

  function stopHeartbeat() {
    if (heartbeatTimer.current !== null) {
      window.clearInterval(heartbeatTimer.current);
      heartbeatTimer.current = null;
    }
  }

  function stopElapsedClock() {
    if (elapsedTimer.current !== null) {
      window.clearInterval(elapsedTimer.current);
      elapsedTimer.current = null;
    }
  }

  function startElapsedClock() {
    stopElapsedClock();
    sessionStartedAt.current = Date.now();
    setElapsedSeconds(0);
    elapsedTimer.current = window.setInterval(() => {
      if (sessionStartedAt.current !== null) {
        setElapsedSeconds(Math.floor((Date.now() - sessionStartedAt.current) / 1000));
      }
      setDroppedAudioSeconds(droppedAudioSecondsRef.current);
    }, 1000);
  }

  function cleanupAudio(preserveReconnectFrames = false) {
    if (reconnectTimer.current !== null) {
      window.clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    stopHeartbeat();
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
    if (!preserveReconnectFrames) reconnectFrames.current.clear();
    pcmSequence.current = 0;
    nextTtsTime.current = 0;
    mediaStream.current?.getTracks().forEach((track) => track.stop());
    mediaStream.current = null;
    void audioContext.current?.close();
    audioContext.current = null;
    analyser.current = null;
    speechStarted.current = false;
    silenceSince.current = null;
    captureSuspendedAt.current = null;
    pendingSpeechEnd.current = false;
  }

  function prepareServerRecording() {
    recordingId.current = crypto.randomUUID();
    recordingConfirmed.current = false;
    finalizeAfterReconnect.current = false;
    reconnectFrames.current.clear();
    droppedAudioSecondsRef.current = 0;
    setDroppedAudioSeconds(0);
    setReconnectCount(0);
    setRecordingStarted(false);
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

  function startHeartbeat() {
    if (heartbeatTimer.current !== null) return;
    lastServerMessageAt.current = Date.now();
    heartbeatTimer.current = window.setInterval(() => {
      const connection = socket.current;
      if (connection?.readyState !== WebSocket.OPEN) return;
      const now = Date.now();
      if (heartbeatExpired(lastServerMessageAt.current, now)) {
        setConnectionError("服务端长时间无响应，正在自动重连…");
        connection.close(4000, "heartbeat timeout");
        return;
      }
      connection.send(JSON.stringify({ type: "heartbeat" }));
    }, HEARTBEAT_INTERVAL_MS);
  }

  function flushReconnectFrames(connection: WebSocket) {
    if (connection.readyState !== WebSocket.OPEN) return;
    for (const frame of reconnectFrames.current.drain()) {
      connection.send(frame);
    }
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
    if (
      stateRef.current === "LISTENING"
      || stateRef.current === "SPEAKING"
      || stateRef.current === "CONNECTING"
    ) {
      if (level >= SPEECH_THRESHOLD) {
        if (!speechStarted.current) {
          speechStarted.current = true;
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
            connection.send(JSON.stringify({ type: "speech_end" }));
            stateRef.current = "REASONING";
            setState("REASONING");
          } else {
            pendingSpeechEnd.current = true;
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
    const stream = mediaStream.current ?? await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStream.current = stream;
    for (const track of stream.getAudioTracks()) {
      track.onended = () => {
        if (!micActive.current) return;
        micActive.current = false;
        setConnectionError("麦克风设备已断开；当前录音可停止保存，重新开始后可选择新设备");
      };
    }
    const context = new AudioContext({ sampleRate: JAEC_SAMPLE_RATE });
    audioContext.current = context;
    // 标签页切后台或系统休眠会让 AudioContext 进入 suspended,onaudioprocess 停止,
    // 录音静默中断。监听 statechange 并尝试自动 resume。
    context.onstatechange = () => {
      if (context.state === "suspended" && micActive.current) {
        captureSuspendedAt.current ??= Date.now();
        void context.resume().catch(() => {
          setConnectionError("浏览器暂停了麦克风；回到页面并点击任意位置后会再次恢复");
        });
      } else if (context.state === "running" && captureSuspendedAt.current !== null) {
        droppedAudioSecondsRef.current += (Date.now() - captureSuspendedAt.current) / 1000;
        captureSuspendedAt.current = null;
        setDroppedAudioSeconds(droppedAudioSecondsRef.current);
        setConnectionError("麦克风采集已恢复；暂停期间已计入未补传缺口");
      }
    };
    micActive.current = true;
    if (connection.readyState !== WebSocket.OPEN) {
      throw new Error("连接在麦克风启动前已断开，正在重连");
    }
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
        const microphoneFrame = encodePcm16Frame(
          microphoneChunks[index], sequence, "microphone", JAEC_SAMPLE_RATE,
        );
        if (
          live?.readyState === WebSocket.OPEN
          && live.bufferedAmount <= MAX_SOCKET_BUFFERED_BYTES
        ) {
          if (reconnectFrames.current.length > 0) flushReconnectFrames(live);
          live.send(microphoneFrame);
        } else if (sessionActiveRef.current) {
          droppedAudioSecondsRef.current += reconnectFrames.current.push(microphoneFrame);
        }
        const reference = referenceChunks[index];
        if (
          reference &&
          live?.readyState === WebSocket.OPEN &&
          live.bufferedAmount <= MAX_SOCKET_BUFFERED_BYTES &&
          stateRef.current === "SPEAKING"
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
    if (
      sessionActiveRef.current
      || socket.current?.readyState === WebSocket.OPEN
      || socket.current?.readyState === WebSocket.CONNECTING
    ) return;
    setMicAccessIssue(null);
    setConnectionError("");
    setState("CONNECTING");
    // Request permission directly from the user's click, before network checks
    // can consume the browser's user-activation window.
    if (!mediaStream.current) {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error("当前浏览器不支持麦克风录音，请使用 HTTPS 或 localhost");
        }
        mediaStream.current = await navigator.mediaDevices.getUserMedia({
          audio: selectedMicDeviceId ? { deviceId: { exact: selectedMicDeviceId } } : true,
        });
        await refreshMicrophoneDevices();
      } catch (error) {
        await refreshMicrophoneDevices();
        const name = error instanceof Error ? error.name : "";
        const issue: MicAccessIssue = name === "NotAllowedError" || name === "SecurityError"
          ? "permission"
          : name === "NotFoundError" || name === "OverconstrainedError"
            ? "device"
            : "unavailable";
        const message = issue === "permission"
          ? "麦克风权限未授予，请在浏览器或系统设置中允许访问后重试。"
          : issue === "device"
            ? "未找到可用的麦克风，请连接或启用输入设备后重试。"
            : error instanceof Error ? error.message : "无法访问麦克风";
        setMicAccessIssue(issue);
        setConnectionError(message);
        setState("IDLE");
        return;
      }
    }
    let pinned = selectedSessionVoice();
    if (selectedVoiceId) {
      try {
        const [voice, health] = await Promise.all([getVoice(selectedVoiceId), getTtsHealth()]);
        if (!sessionSelectableVoices([voice], health).length || !pinned) {
          cleanupAudio();
          setSelectedVoiceId("");
          setState("IDLE");
          setConnectionError("所选音色已撤销授权或引擎不可用，已切换为系统默认 TTS。请再次开始通话。");
          return;
        }
        // Re-read the revision immediately before the handshake so an update
        // between list rendering and clicking "开始通话" cannot pin stale data.
        pinned = {
          voiceId: voice.voice_id,
          revision: voice.revision!.revision,
          label: voice.display_name,
          backend: voice.revision!.backend_family,
        };
      } catch {
        cleanupAudio();
        setState("IDLE");
        setConnectionError("无法确认所选音色的最新状态；请稍后重试或切换系统默认 TTS。");
        return;
      }
    }
    manualStop.current = false;
    reconnectAttempts.current = 0;
    sessionVoice.current = pinned;
    prepareServerRecording();
    setSessionIsActive(true);
    setTranscript("");
    setAnswer("");
    setStopping(false);
    startElapsedClock();
    stateRef.current = "CONNECTING";
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
      if (socket.current !== connection) {
        connection.close();
        return;
      }
      // 服务端为每个 WebSocket 创建一个新的 Session，generation 会从 0 开始。
      // 因此重连时必须丢弃旧连接的代际水位，不能把新事件当作过期事件过滤。
      activeGeneration.current = generationForNewConnection();
      setConnected(true);
      reconnectAttempts.current = 0;
      lastServerMessageAt.current = Date.now();
      startHeartbeat();
      connection.send(
        JSON.stringify(startSessionPayload(recordingId.current, sessionVoice.current)),
      );
      if (finalizeAfterReconnect.current) {
        connection.send(
          JSON.stringify({
            type: "audio_format",
            format: "pcm16",
            sample_rate: JAEC_SAMPLE_RATE,
            frame_samples: PCM_FRAME_SAMPLES,
          }),
        );
        flushReconnectFrames(connection);
        connection.send(JSON.stringify({ type: "end_session" }));
        setConnectionError("连接已恢复，正在整理服务端录音…");
        return;
      }
      if (startMic && !micActive.current) {
        void startMicrophone(connection)
          .then(() => setConnectionError("麦克风已连接，说话后自动提交"))
          .catch((error: unknown) => {
            const name = error instanceof Error ? error.name : "";
            const issue: MicAccessIssue = name === "NotAllowedError" || name === "SecurityError"
              ? "permission"
              : name === "NotFoundError" || name === "OverconstrainedError"
                ? "device"
                : "unavailable";
            setMicAccessIssue(issue);
            setConnectionError(error instanceof Error ? error.message : "无法访问麦克风");
            manualStop.current = true;
            setSessionIsActive(false);
            stopElapsedClock();
            cleanupAudio();
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
        flushReconnectFrames(connection);
        if (pendingSpeechEnd.current) {
          pendingSpeechEnd.current = false;
          connection.send(JSON.stringify({ type: "speech_end" }));
          stateRef.current = "REASONING";
          setState("REASONING");
          setConnectionError("连接已恢复，正在转写断线期间缓存的语音");
        } else {
          stateRef.current = "LISTENING";
          setState("LISTENING");
          setConnectionError(
            droppedAudioSecondsRef.current > 0
              ? `连接已恢复；有 ${droppedAudioSecondsRef.current.toFixed(1)} 秒超出本地缓存`
              : "连接已恢复，缓存语音已补传",
          );
        }
      }
    };
    connection.onclose = () => {
      if (socket.current !== connection) return;
      socket.current = null;
      setConnected(false);
      if (manualStop.current) {
        cleanupAudio();
        setStopping(false);
        stateRef.current = "IDLE";
        setState("IDLE");
        sessionVoice.current = null;
        return;
      }
      // 意外断连:保留麦克风与录音,自动重连(指数退避,最长 10s)。
      setReconnectCount((current) => current + 1);
      scheduleReconnect();
    };
    connection.onerror = () => {
      if (!manualStop.current) {
        setConnectionError("连接中断，正在尝试重连…");
      }
    };
    connection.onmessage = (message) => {
      if (socket.current !== connection) return;
      lastServerMessageAt.current = Date.now();
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
    if (event.type === "asr_final" && event.text) {
      setTranscript((current) => {
        const combined = current ? `${current}\n${event.text}` : event.text ?? "";
        return combined.slice(-MAX_TRANSCRIPT_CHARACTERS);
      });
    }
    if (event.type === "llm_token") setAnswer((current) => current + (event.token ?? ""));
    if (event.type === "tts_chunk" && event.audio_b64) {
      void playTtsChunk(event);
    }
    if (event.type === "tts_end") {
      stateRef.current = "LISTENING";
      setState("LISTENING");
    }
    if (event.type === "recording_started" && event.recording_id) {
      recordingId.current = event.recording_id;
      recordingConfirmed.current = true;
      setRecordingStarted(true);
    }
    if (event.type === "recording_ready") {
      if (event.download_url) {
        setRecordingDownloadUrl(event.download_url);
        setRecordingReady(true);
        setConnectionError("录音已整理完成，可下载到本地");
      } else {
        setConnectionError("本次通话没有可导出的音频");
      }
      finalizeAfterReconnect.current = false;
      manualStop.current = true;
      recordingConfirmed.current = false;
      setRecordingStarted(false);
      setSessionIsActive(false);
      stopElapsedClock();
      setDroppedAudioSeconds(droppedAudioSecondsRef.current);
      socket.current?.close();
    }
    if (
      event.type === "error"
      && (event.code === "audio_segment_limit_reached" || event.code === "audio_too_large")
    ) {
      const connection = socket.current;
      if (connection?.readyState === WebSocket.OPEN && stateRef.current === "LISTENING") {
        connection.send(JSON.stringify({ type: "speech_end" }));
        stateRef.current = "REASONING";
        setState("REASONING");
      }
      setConnectionError("单段语音较长，已自动提交当前段；完整录音仍在继续保存");
    }
    if (event.type === "error" && event.code === "recording_too_large") {
      const maxMiB = Math.round((event.max_bytes ?? 0) / 1024 / 1024);
      setConnectionError(`服务端录音达到 ${maxMiB} MiB 上限；实时通话继续进行`);
    }
    if (event.type === "error" && event.message && event.code !== "recording_too_large") {
      setConnectionError(event.message);
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
    setStopping(true);
    const hadCapturedAudio = micActive.current
      || reconnectFrames.current.length > 0
      || recordingConfirmed.current;
    if (!hadCapturedAudio) {
      manualStop.current = true;
      finalizeAfterReconnect.current = false;
      socket.current?.close();
      cleanupAudio();
      setConnected(false);
      setSessionIsActive(false);
      stopElapsedClock();
      setStopping(false);
      stateRef.current = "IDLE";
      setState("IDLE");
      setConnectionError("已取消连接");
      return;
    }

    finalizeAfterReconnect.current = true;
    cleanupAudio(true);
    const connection = socket.current;
    if (connection?.readyState === WebSocket.OPEN) {
      flushReconnectFrames(connection);
      // In LISTENING, the server may still hold the final utterance waiting
      // for VAD silence. Let end_session flush it through ASR. Interrupt only
      // an already-running reply, otherwise a previous generation would move
      // the session to IDLE and discard the pending microphone audio.
      if (stateRef.current === "REASONING" || stateRef.current === "SPEAKING") {
        connection.send(
          JSON.stringify({
            type: "interrupt",
            generation_id: activeGeneration.current || undefined,
            reason: "manual",
          }),
        );
      }
      connection.send(JSON.stringify({ type: "end_session" }));
      setConnectionError("正在将服务端 PCM 录音整理为 WAV…");
      return;
    }
    if (connection?.readyState === WebSocket.CONNECTING) {
      setConnectionError("连接恢复后将自动整理录音，请勿关闭页面");
      return;
    }
    setConnectionError("正在重新连接服务端以整理录音，请勿关闭页面");
    openSocket(false);
  }

  const listening = state === "LISTENING";
  const centerLabel = stateLabels[state] ?? state;
  const connectionPhase = stopping
    ? "Finalizing"
    : connected
      ? "Live"
      : sessionActive
        ? "Reconnecting"
        : "Standby";

  return (
    <section className="view" aria-label="实时助手">
      {micAccessIssue && (
        <div className="mic-dialog-backdrop" role="presentation" onMouseDown={(event) => {
          if (event.target === event.currentTarget) setMicAccessIssue(null);
        }}>
          <section className="mic-dialog" role="dialog" aria-modal="true" aria-labelledby="mic-dialog-title">
            <div className="mic-dialog__icon" aria-hidden="true">♩</div>
            <h2 id="mic-dialog-title">
              {micAccessIssue === "permission" ? "需要麦克风权限" : micAccessIssue === "device" ? "没有找到麦克风" : "无法访问麦克风"}
            </h2>
            <p>
              {micAccessIssue === "permission"
                ? "Astra 需要麦克风才能进行实时语音对话。点击下方按钮重新请求权限；如果浏览器已阻止，请在地址栏的网站设置或 macOS“系统设置 → 隐私与安全性 → 麦克风”中允许当前浏览器。"
                : micAccessIssue === "device"
                  ? "请连接麦克风，并在 macOS“系统设置 → 声音 → 输入”中确认设备可用，然后重试。"
                  : "请确认浏览器拥有麦克风权限，且输入设备没有被其他应用独占，再重新尝试。"}
            </p>
            <div className="mic-dialog__actions">
              <button className="btn btn--secondary" type="button" onClick={() => setMicAccessIssue(null)}>稍后处理</button>
              <button className="btn btn--primary" type="button" onClick={() => void connect()}>
                {micAccessIssue === "permission" ? "重新请求权限" : "重试连接"}
              </button>
            </div>
          </section>
        </div>
      )}
      <div className="page-head">
        <div className="page-head__text">
          <div className="eyebrow">01 · 实时助手</div>
          <h1>持续录音，停顿后自动转写</h1>
          <p>录音持续写入 Mac mini；短暂断线会在浏览器内缓存音频并自动补传。</p>
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
            <label className="live-voice-select">
              <span>回复音色</span>
              <select
                className="select"
                value={selectedVoiceId}
                disabled={connected || stopping}
                onChange={(event) => setSelectedVoiceId(event.target.value)}
                aria-describedby="live-voice-hint"
              >
                <option value="">系统默认（{ttsHealth?.default_backend || "当前引擎"}）</option>
                {voiceChoices.map((voice) => (
                  <option value={voice.voice_id} key={voice.voice_id}>
                    {voice.display_name} · r{voice.revision?.revision} · {voice.revision?.backend_family}
                  </option>
                ))}
              </select>
              <span className="field__hint" id="live-voice-hint">
                {connected
                  ? "会话进行中，音色版本已锁定。"
                  : voiceChoices.length
                    ? "仅显示已授权、可用且通过设备验证的音色。"
                    : "暂无可用克隆音色；可继续使用系统默认 TTS。"}
              </span>
            </label>
            <label className="live-voice-select">
              <span>麦克风输入</span>
              <select
                className="select"
                value={selectedMicDeviceId}
                disabled={connected || stopping || state === "CONNECTING"}
                onChange={(event) => setSelectedMicDeviceId(event.target.value)}
                aria-describedby="live-mic-hint"
              >
                <option value="">系统默认设备</option>
                {microphoneDevices.map((device, index) => (
                  <option value={device.deviceId} key={`${device.deviceId}-${index}`}>
                    {device.label || `麦克风 ${index + 1}`}
                  </option>
                ))}
              </select>
              <span className="field__hint" id="live-mic-hint">
                {microphoneDevices.length
                  ? "已检测到输入设备；首次使用时浏览器会请求麦克风权限。"
                  : "尚未检测到输入设备；允许麦克风权限后可重新检测。"}
              </span>
            </label>
            {!sessionActive ? (
              <button className="btn btn--primary btn--sm" type="button" onClick={connect}>
                开始通话
              </button>
            ) : (
              <button className="btn btn--danger btn--sm" type="button" onClick={stop} disabled={stopping}>
                {stopping
                  ? "正在保存…"
                  : recordingStarted
                    ? "停止并保存"
                    : "取消连接"}
              </button>
            )}
            <button
              className="btn btn--secondary btn--sm"
              type="button"
              onClick={saveLocalRecording}
              disabled={!recordingReady || sessionActive}
              title={sessionActive ? "停止并完成服务端录音后下载" : "下载服务端保存的本次通话录音"}
            >
              下载录音
            </button>
          </div>
          <div className="dial-meta">
            <div>
              <div className="dial-meta__k">会话状态</div>
              <div className="dial-meta__v">{connectionPhase}</div>
            </div>
            <div>
              <div className="dial-meta__k">当前阶段</div>
              <div className="dial-meta__v">{centerLabel}</div>
            </div>
            <div>
              <div className="dial-meta__k">录音时长</div>
              <div className="dial-meta__v">{formatElapsedTime(elapsedSeconds)}</div>
            </div>
            <div>
              <div className="dial-meta__k">输入设备</div>
              <div className="dial-meta__v">{microphoneDevices.find((device) => device.deviceId === selectedMicDeviceId)?.label || "系统默认输入"}</div>
            </div>
            <div>
              <div className="dial-meta__k">自动重连</div>
              <div className="dial-meta__v">{reconnectCount} 次</div>
            </div>
            <div>
              <div className="dial-meta__k">未补传缺口</div>
              <div className="dial-meta__v">
                {droppedAudioSeconds > 0 ? `${droppedAudioSeconds.toFixed(1)} s` : "无"}
              </div>
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
                {connected ? "录音中" : sessionActive ? "恢复中" : "未连接"}
              </span>
            </div>
            <div className="live-voice-summary">
              <span>当前回复</span>
              <strong>{sessionVoice.current ? `${sessionVoice.current.label} · ${sessionVoice.current.backend} · r${sessionVoice.current.revision}` : `系统默认 · ${ttsHealth?.default_backend || "当前引擎"}`}</strong>
              {connected && <span className="badge badge--accent">版本已锁定</span>}
            </div>
            <div className="spine">
              <div className="spine-row">
                <div className="spine-time">你</div>
                <div className="spine-body">
                  <div className="spine-speaker">语音输入</div>
                  <div className={`spine-text${transcript ? "" : " spine-text--dim"}`}>
                    {transcript || "等待语音输入。转写结果会在本次会话中持续累积。"}
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

          <div className="alert alert--info" role="status" aria-live="polite">
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
