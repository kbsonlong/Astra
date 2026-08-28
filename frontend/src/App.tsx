import { useEffect, useRef, useState } from "react";

type ServerEvent = {
  type: string;
  state?: string;
  text?: string;
  token?: string;
  generation_id?: number;
  audio_b64?: string;
  mime?: string;
};

const socketUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

export default function App() {
  const socket = useRef<WebSocket | null>(null);
  const activeGeneration = useRef(0);
  const [state, setState] = useState("IDLE");
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [connected, setConnected] = useState(false);

  useEffect(() => () => socket.current?.close(), []);

  function connect() {
    if (socket.current?.readyState === WebSocket.OPEN) return;
    const connection = new WebSocket(socketUrl);
    connection.onopen = () => {
      setConnected(true);
      connection.send(JSON.stringify({ type: "start_session" }));
    };
    connection.onclose = () => {
      setConnected(false);
      setState("IDLE");
    };
    connection.onmessage = (message) => handleEvent(JSON.parse(message.data) as ServerEvent);
    socket.current = connection;
  }

  function handleEvent(event: ServerEvent) {
    if (event.generation_id && event.type !== "state_change") {
      if (event.generation_id < activeGeneration.current) return;
      activeGeneration.current = event.generation_id;
    }
    if (event.type === "state_change" && event.state) setState(event.state);
    if (event.type === "asr_final") setTranscript(event.text ?? "");
    if (event.type === "llm_token") setAnswer((current) => current + (event.token ?? ""));
    if (event.type === "tts_chunk" && event.audio_b64) {
      const bytes = Uint8Array.from(atob(event.audio_b64), (char) => char.charCodeAt(0));
      new Audio(URL.createObjectURL(new Blob([bytes], { type: event.mime ?? "audio/wav" }))).play();
    }
    if (event.type === "tts_end") setState("LISTENING");
  }

  function stop() {
    socket.current?.send(JSON.stringify({
      type: "interrupt",
      generation_id: activeGeneration.current || undefined,
      reason: "manual",
    }));
    socket.current?.send(JSON.stringify({ type: "end_session" }));
    socket.current?.close();
    setConnected(false);
    setState("IDLE");
  }

  return (
    <main className="shell">
      <header>
        <span className="eyebrow">ASTRA / LOCAL VOICE</span>
        <h1>语音助手</h1>
        <span className={`status ${connected ? "online" : "offline"}`}>{state}</span>
      </header>
      <section className="conversation" aria-live="polite">
        <div className="message user"><span>你</span><p>{transcript || "等待语音输入"}</p></div>
        <div className="message assistant"><span>Astra</span><p>{answer || "准备好后开始对话"}</p></div>
      </section>
      <footer>
        {!connected ? <button onClick={connect}>开始通话</button> : <button className="stop" onClick={stop}>停止</button>}
        <small>{connected ? "连接已建立" : "尚未连接"}</small>
      </footer>
    </main>
  );
}
