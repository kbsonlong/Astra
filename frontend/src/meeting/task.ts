export type OverlapDetection = {
  status?: "ok" | "failed";
  suspected: boolean;
  score: number;
  active_frames: number;
  candidate_frames: number;
  threshold: number;
  details?: {
    sample_rate?: number;
    frame_seconds?: number;
    hop_seconds?: number;
    frequency_band_hz?: number[];
    eligible_frames?: number;
  };
};

export type MeetingResult = {
  task_id: string;
  filename: string;
  duration_s: number;
  language: string;
  segments: number;
  speakers: string[];
  report_path: string;
  summary: string;
  timeline_preview: string;
  prompt_template?: string;
  training_artifacts?: {
    clips_dir?: string;
    transcript_segments?: string;
    training_candidates?: string;
    candidate_segments?: number;
  };
  separation_audio_urls?: string[];
  overlap_detection?: OverlapDetection;
};

export type MeetingStatus = {
  type?: string;
  task_id: string;
  status: "processing" | "done" | "failed" | "stopped";
  duration_s?: number;
  segments?: number;
  speakers?: string[];
  report_path?: string;
  summary_preview?: string;
  transcript_preview?: string;
  elapsed_s?: number;
  error?: string;
  message?: string;
  training_artifacts?: MeetingResult["training_artifacts"];
  prompt_template?: string;
  separation_audio_urls?: string[];
  overlap_detection?: OverlapDetection;
};

export function meetingResultFromStatus(
  current: MeetingStatus,
  filename: string,
  reportPath: string,
): MeetingResult {
  return {
    task_id: current.task_id,
    filename,
    duration_s: current.duration_s ?? 0,
    language: "Chinese",
    segments: current.segments ?? 0,
    speakers: current.speakers ?? [],
    report_path: current.report_path ?? reportPath,
    summary: current.summary_preview ?? "",
    timeline_preview: current.transcript_preview ?? "",
    training_artifacts: current.training_artifacts,
    prompt_template: current.prompt_template,
    separation_audio_urls: current.separation_audio_urls,
    overlap_detection: current.overlap_detection,
  };
}

export const meetingEventUrl = (taskId: string): string => {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${location.host}/api/meeting/${encodeURIComponent(taskId)}/events`;
};

type SocketRef = { current: WebSocket | null };

export function waitForMeeting(
  taskId: string,
  reportPath: string,
  filename: string,
  socketRef: SocketRef,
  setStatus: (status: string) => void,
): Promise<MeetingResult> {
  socketRef.current?.close();
  return new Promise((resolve, reject) => {
    let finished = false;
    let reconnectAttempts = 0;
    let reconnectTimer: number | undefined;

    const complete = (result: MeetingResult) => {
      if (finished) return;
      finished = true;
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socketRef.current?.close();
      socketRef.current = null;
      resolve(result);
    };

    const consume = (current: MeetingStatus) => {
      if (current.type === "error") {
        finished = true;
        socketRef.current?.close();
        socketRef.current = null;
        reject(new Error(current.message ?? "会议状态订阅失败"));
        return;
      }
      if (current.status === "processing") {
        reconnectAttempts = 0;
        setStatus(`会议处理中 ${current.elapsed_s ?? 0}s`);
        return;
      }
      if (current.status === "failed" || current.status === "stopped") {
        finished = true;
        socketRef.current?.close();
        socketRef.current = null;
        reject(new Error(current.error ?? (current.status === "stopped" ? "会议已取消" : "会议处理失败，请查看服务端日志")));
        return;
      }
      complete(meetingResultFromStatus(current, filename, reportPath));
    };

    const scheduleRecovery = () => {
      if (finished || reconnectTimer !== undefined) return;
      const delay = Math.min(5000, 500 * 2 ** reconnectAttempts);
      reconnectAttempts += 1;
      setStatus("会议状态连接中断，正在恢复…");
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = undefined;
        void recover();
      }, delay);
    };

    const connect = () => {
      if (finished) return;
      const connection = new WebSocket(meetingEventUrl(taskId));
      socketRef.current = connection;
      connection.onmessage = (message) => {
        try {
          consume(JSON.parse(message.data) as MeetingStatus);
        } catch {
          finished = true;
          connection.close();
          reject(new Error("会议状态 WebSocket 返回了无法识别的消息"));
        }
      };
      connection.onerror = () => {
        // onclose schedules HTTP recovery; onerror alone is not terminal.
      };
      connection.onclose = () => {
        if (socketRef.current === connection) socketRef.current = null;
        if (!finished) scheduleRecovery();
      };
    };

    async function recover() {
      if (finished) return;
      try {
        const response = await fetch(`/api/meeting/${encodeURIComponent(taskId)}`);
        const payload = await response.json() as MeetingStatus | { detail?: string };
        if (!response.ok) throw new Error("detail" in payload ? payload.detail : "会议状态暂不可用");
        consume(payload as MeetingStatus);
        if (!finished && (payload as MeetingStatus).status === "processing") connect();
      } catch {
        scheduleRecovery();
      }
    }

    connect();
  });
}
