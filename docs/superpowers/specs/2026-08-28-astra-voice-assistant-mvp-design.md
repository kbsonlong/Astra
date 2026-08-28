# Astra 语音助手 MVP - 设计文档

## 1. 项目概述

### 1.1 项目背景

基于 Mac mini 16G（Apple Silicon）构建一套局域网内可用的实时语音通话助手。MVP 阶段优先实现 Web 端实时语音通话，后续再扩展翻译助手、桌面端和移动端。

本方案采用分层部署：

- Mac mini 本机只承载 Web、FastAPI 网关、ASR、TTS。
- LLM 推理复用局域网机器 `192.168.3.18`，由 `omlx` 或 `llama-server` 在该机器 host 原生运行。
- FastAPI 只依赖 `192.168.3.18` 暴露的 OpenAI 兼容 HTTP 协议，不感知底层 LLM 运行时。

因此本文档不再表述为“完全本地化”，而是“Mac mini 本地 ASR/TTS + 局域网 LLM”。

### 1.2 目标用户

| 用户 | 说明 |
|---|---|
| 初期用户 | 开发者 / 本地 AI 爱好者，在同一局域网内通过浏览器使用 |
| 扩展用户 | 日常办公场景个人用户，通过桌面端 / 移动端使用 |

### 1.3 关键约束

| 约束项 | 说明 |
|---|---|
| 硬件底座 | Mac mini 16G，Apple Silicon M 系列 |
| Mac mini 部署 | Docker Compose 启动 4 个容器：`whisper-asr`、`piper-tts`、`fastapi`、`frontend-nginx` |
| LLM 部署 | `192.168.3.18` host 原生运行 `omlx` 或 `llama-server`，不放入 Mac mini Docker Compose |
| MVP 平台 | Web 页面，优先 Chrome / Safari |
| 后端迁移路径 | MVP 使用 Python FastAPI，长期可替换为 Go 网关 |
| 模型服务约束 | Docker Desktop on Mac 不假设 GPU/MPS/ANE 透传；容器内 ASR/TTS 默认走 CPU 可验证路径 |
| 第三方组件 | Redis/MySQL 等如后续需要，统一使用 Docker，不直接安装在 Mac mini 宿主机 |

### 1.4 评审约束落实

| 评审点 | 落实方式 |
|---|---|
| Docker Desktop on Mac 不适合作为 Metal 推理容器 | ASR 默认使用 whisper.cpp Docker CPU；如需 Metal，改为 host 原生 whisper.cpp，通过 `ASR_ENDPOINT` 切换 |
| Qwen3-ASR Apple Silicon Docker 流式路径不可验证 | ASR 切换为 whisper.cpp `ggml-medium`，MVP 只要求整段转写 `asr_final` |
| LLM 端点混乱 / 取消语义不清 | 上层只调用 OpenAI 兼容 `POST /v1/chat/completions`，stream=true；`/api/generate`、自定义 `/cancel` 不进入 MVP 协议 |
| Piper TTS 契约不清晰 | 固定为 HTTP `POST /synthesize`，返回 WAV 22050Hz mono 16-bit；MVP 做句子级合成 |

---

## 2. 总体架构

### 2.1 架构总览

```text
Browser
  |
  | WebSocket + HTTP
  v
Mac mini Docker Compose
  |
  |-- frontend-nginx   React 静态资源 + /api /ws 反代
  |-- fastapi          会话状态机、ASR/LLM/TTS 编排、打断控制
  |-- whisper-asr      whisper.cpp server，默认 Docker CPU
  |-- piper-tts        Piper HTTP server，句子级 WAV 合成
  |
  | HTTP: POST /v1/chat/completions
  v
192.168.3.18
  |
  |-- Profile A: omlx OpenAI-compatible server
  |-- Profile B: llama-server OpenAI-compatible server
```

### 2.2 部署位置

| 组件 | 部署位置 | 说明 |
|---|---|---|
| frontend-nginx | Mac mini Docker | React 静态页面，反代 `/api` 和 `/ws` |
| fastapi | Mac mini Docker | 语音会话网关，不承载 LLM 模型 |
| whisper-asr | Mac mini Docker，默认 | whisper.cpp `ggml-medium`，CPU 路径可验收 |
| whisper-asr | Mac mini host，备选 | 需要 Metal 时启用，FastAPI 改 `ASR_ENDPOINT` |
| piper-tts | Mac mini Docker | Piper HTTP server |
| LLM | `192.168.3.18` host | `omlx` 或 `llama-server`，必须暴露 OpenAI 兼容接口 |

### 2.3 LLM 运行时 Profile

`192.168.3.18` 允许两种运行 Profile。Astra 上层代码只依赖统一协议，不根据运行时写分支逻辑。

| Profile | 运行方式 | 必须满足的协议 |
|---|---|---|
| `omlx` | 在 `192.168.3.18` host 原生运行模型 | 暴露 `POST /v1/chat/completions`，支持 `stream=true` 的 SSE 增量输出；暴露 `GET /v1/models` 或等价探活端点 |
| `llama-server` | 在 `192.168.3.18` host 原生运行 llama.cpp server | 暴露 `POST /v1/chat/completions`，支持 `stream=true` 的 SSE 增量输出；暴露 `GET /v1/models` 或等价探活端点 |

FastAPI 使用以下配置：

```env
LLM_BASE_URL=http://192.168.3.18:8000/v1
LLM_CHAT_PATH=/chat/completions
LLM_MODELS_PATH=/models
LLM_MODEL=<remote-model-name>
LLM_API_KEY=<optional-api-key>
LLM_REQUEST_TIMEOUT_SECONDS=120
LLM_CONNECT_TIMEOUT_SECONDS=3
LLM_STREAM_IDLE_TIMEOUT_SECONDS=15
```

验收只认 OpenAI 兼容协议：

- `GET ${LLM_BASE_URL}${LLM_MODELS_PATH}` 返回 2xx。
- `POST ${LLM_BASE_URL}${LLM_CHAT_PATH}` 支持非流式最小请求。
- `POST ${LLM_BASE_URL}${LLM_CHAT_PATH}` 支持 `stream=true`，响应为 SSE 增量 token。
- 如果某个 `omlx` 启动方式不直接支持 OpenAI 兼容协议，需要在 `192.168.3.18` 上增加适配层；Mac mini FastAPI 不做运行时私有协议适配。

### 2.4 网络策略

FastAPI 容器访问 `192.168.3.18` 使用 Docker Compose 默认 bridge 网络直接访问局域网 IP，不使用 `network_mode: host`，也不使用 `host-gateway`。

约束：

- `192.168.3.18` 的 LLM 服务监听 `0.0.0.0:<port>` 或该机器局域网网卡 IP。
- Mac mini 到 `192.168.3.18:<port>` 必须可达。
- LLM 服务只允许局域网访问，不暴露公网。
- 如配置 API key，FastAPI 从环境变量读取，前端只展示掩码，不把 key 存入 localStorage。

---

## 3. 实时语音状态机

### 3.1 状态定义

| 状态 | 说明 | 触发条件 |
|---|---|---|
| `IDLE` | 空闲，等待用户发起会话 | 初始状态 / 用户结束 / 手动停止 |
| `LISTENING` | 麦克风监听中，前端 VAD 检测句末或插话 | 用户点击开始 / 自动打断后回到监听 |
| `REASONING` | ASR 整段转写 + LLM 流式生成 | 前端发送 `speech_end` |
| `SPEAKING` | TTS 句子级播放回复 | LLM 输出达到句子边界并完成首句 TTS |

### 3.2 状态转移

```text
IDLE -- start_session --> LISTENING
LISTENING -- speech_end --> REASONING
REASONING -- first_tts_chunk --> SPEAKING
REASONING -- interrupt/manual_stop --> LISTENING or IDLE
SPEAKING -- vad_interrupt --> LISTENING
SPEAKING -- manual_stop --> IDLE
SPEAKING -- tts_end --> LISTENING
LISTENING -- end_session --> IDLE
```

### 3.3 generation_id 规则

- 每次 `speech_end` 生成一个新的 `generation_id`。
- 后端所有 ASR、LLM、TTS 任务按 `(session_id, generation_id)` 建索引。
- 前端收到任何带 `generation_id` 的消息时，只处理当前活跃轮次。
- 中断后旧 `generation_id` 的 token、TTS 音频、结束事件全部丢弃。

---

## 4. 打断与取消设计

### 4.1 VAD 自动打断

自动打断只在 `SPEAKING` 阶段生效。

1. 前端以约 100Hz 分析麦克风 RMS。
2. 连续 300ms 超过阈值，触发 `vad_interrupt`。
3. 前端 300ms 内执行 `audioPlayer.flush()`，停止当前和已排队音频。
4. 前端发送 `interrupt`，携带当前 `generation_id`。
5. 后端按 `(session_id, generation_id)` 命中任务组并取消。
6. 状态切回 `LISTENING`，麦克风保持开启。

### 4.2 手动停止

手动停止在任意状态生效。

1. 前端停止录音和播放。
2. 前端发送 `interrupt`。
3. 后端取消命中的 ASR/LLM/TTS 任务。
4. 状态切回 `IDLE`。

### 4.3 后端取消链路

后端取消动作必须按以下顺序执行：

1. 标记 `generation_id` 为 cancelled，阻止后续消息入队。
2. 关闭 LLM SSE HTTP 响应：`await response.aclose()`。
3. 取消 `llm_task`、`sentence_split_task`、`tts_task`。
4. 关闭正在执行的 Piper HTTP 请求。
5. 清空该 `generation_id` 对应的 TTS 待发送队列。
6. 向前端发送 `state_change`。

### 4.4 “真停”验收口径

LLM 在 `192.168.3.18` 运行，Mac mini 只能证明本地链路停止，不能仅靠关闭 SSE 证明远端推理进程一定停止。因此 MVP 将“真停”拆成两个层次：

| 层次 | 验收要求 |
|---|---|
| Mac mini 本地真停 | interrupt 后 300ms 内前端静音；500ms 内 FastAPI 不再向前端发送旧 `generation_id` 消息；2s 内 `whisper-asr` / `piper-tts` CPU 回落到 idle + 5% 以内 |
| 远端 LLM 停止 | interrupt 后 2s 内 FastAPI 无旧 token 流入；`192.168.3.18` 服务日志记录 client disconnected 或 request aborted；如远端运行时暴露指标，则 CPU/GPU 活跃度在 2s 内回落 |

若 `omlx` 或 `llama-server` 版本不能提供远端 request abort 可观测性，验收结论必须写为“远端 LLM best-effort cancel 已触发，缺少运行时级停止证明”，不能宣称远端推理已被严格终止。

---

## 5. WebSocket 消息协议

### 5.1 Client -> Server

| 类型 | 数据 | 说明 |
|---|---|---|
| `start_session` | `{}` | 发起会话 |
| `audio_chunk` | binary PCM | 16kHz 16-bit mono，建议 20ms 一包 |
| `speech_end` | `{ generation_id: number }` | 前端 VAD 检测到 800ms 静音后提交句末 |
| `interrupt` | `{ generation_id: number, reason: "vad" \| "manual" }` | 取消指定轮次 |
| `end_session` | `{}` | 结束会话 |

### 5.2 Server -> Client

| 类型 | 数据 | 说明 |
|---|---|---|
| `state_change` | `{ state, generation_id?: number }` | 状态变更 |
| `asr_final` | `{ text: string, generation_id: number }` | ASR 最终结果 |
| `asr_partial` | `{ text: string, generation_id: number }` | 可选，MVP 不强制 |
| `llm_token` | `{ token: string, generation_id: number }` | LLM SSE token |
| `tts_start` | `{ generation_id: number, seq: number }` | 某句 TTS 开始 |
| `tts_chunk` | `{ generation_id: number, seq: number, mime: "audio/wav", sample_rate: 22050, audio_b64: string }` | TTS 音频分片，前端按 `generation_id` 和 `seq` 校验 |
| `tts_end` | `{ generation_id: number }` | 当前轮次播放结束 |
| `error` | `{ code: string, message: string, generation_id?: number }` | 错误 |

`tts_chunk` 不使用裸二进制帧，避免中断后旧音频无法按轮次丢弃。MVP 先使用 base64 JSON，后续若性能不足，再改为“JSON header + binary frame”的双帧协议。

---

## 6. MVP 功能边界

### 6.1 包含功能

| 模块 | 功能 |
|---|---|
| 语音通话 | 开始 / 结束通话、麦克风录音、波形展示 |
| VAD | 前端音量阈值 VAD，支持句末提交和说话打断 |
| ASR | whisper.cpp `ggml-medium` 整段转写，至少输出 `asr_final` |
| LLM | 调用 `192.168.3.18` OpenAI 兼容 SSE，逐 token 展示 |
| TTS | Piper HTTP 句子级合成与播放，可随时打断 |
| 上下文 | 最近 10 轮对话送入 LLM |
| 配置 | LLM base URL、model、temperature、top_p、system prompt、ASR/TTS endpoint、VAD 阈值 |
| 部署 | Docker Compose 启动 Mac mini 4 容器，LLM 在 `192.168.3.18` 外部运行 |

### 6.2 不包含功能

- 会话持久化。
- 历史会话列表。
- 多用户鉴权。
- 多模型热切换。
- Tauri 桌面端。
- iOS / Android 移动端。
- 翻译助手。
- token 级 TTS。
- ASR 字级实时 partial。

---

## 7. 技术选型

### 7.1 前端

| 组件 | 选型 |
|---|---|
| 框架 | React 18 + TypeScript + Vite |
| 状态管理 | Zustand |
| UI | shadcn/ui + Tailwind CSS |
| 录音 | MediaRecorder API + Web Audio API |
| VAD | 自研 RMS 阈值检测，后续可替换 Silero VAD WASM |
| 播放 | Web Audio API AudioContext + 队列 |
| 通信 | 原生 WebSocket，30s 心跳 |

### 7.2 后端

| 组件 | 选型 | 说明 |
|---|---|---|
| 框架 | Python 3.11 + FastAPI | HTTP API + WebSocket |
| 会话管理 | 内存 dict + asyncio.Lock | MVP 单机内存态 |
| 流水线 | asyncio.TaskGroup + cancellation token | ASR -> LLM -> sentence splitter -> TTS |
| LLM 客户端 | `OpenAICompatLLMClient` | 只调用 `/v1/chat/completions` 和 `/v1/models` |
| ASR 客户端 | `WhisperCppAsrClient` | 调 whisper.cpp HTTP `/inference` |
| TTS 客户端 | `PiperHttpTtsClient` | 调 Piper HTTP `/synthesize` |

### 7.3 模型服务

| 服务 | 选型 | 模型 | 资源口径 |
|---|---|---|---|
| LLM | `192.168.3.18` 上的 `omlx` 或 `llama-server` | 由远端机器决定 | 不计入 Mac mini Docker RSS |
| ASR | whisper.cpp server | `ggml-medium.bin` | 容器 RSS 目标 0.75-1.2GB |
| TTS | Piper HTTP server | `zh_CN` + `en_US` 音色 | 容器 RSS 目标 0.3-0.6GB |
| FastAPI | Python 3.11 slim | 无 | 容器 RSS 目标 0.2-0.8GB |
| Nginx | nginx alpine | 无 | 容器 RSS 可忽略 |

---

## 8. 部署设计

### 8.1 Docker Compose 服务

Compose 只包含 Mac mini 本地组件：

- `frontend-nginx`
- `fastapi`
- `whisper-asr`
- `piper-tts`

不包含：

- Ollama。
- `omlx`。
- `llama-server`。
- 任何 LLM 模型下载或 LLM 运行时。

### 8.2 镜像与模型锁定

实现阶段需要锁定以下内容，不能使用裸 `latest` 作为最终交付：

| 项 | 要求 |
|---|---|
| whisper.cpp 镜像 | 固定 tag 或 digest |
| Piper HTTP server 镜像 | 固定 tag 或 digest，并在 README 写明具体仓库 |
| `ggml-medium.bin` | 写明下载 URL、文件名、SHA256 |
| Piper voice | 写明音色文件、下载 URL、SHA256 |
| Python 依赖 | `requirements.txt` 固定主版本，关键库固定精确版本 |
| 前端依赖 | lockfile 纳入仓库 |

### 8.3 健康检查

| 服务 | 健康检查 |
|---|---|
| `whisper-asr` | HTTP health 或最小 WAV 转写探针；必须证明模型已加载 |
| `piper-tts` | HTTP health 或最小文本合成探针；必须证明音色已加载 |
| `fastapi` | `GET /api/health`，聚合 LLM/ASR/TTS 状态 |
| `frontend-nginx` | `GET /` 返回 200 |
| `192.168.3.18` LLM | `GET /v1/models` 返回 2xx，流式 chat smoke test 通过 |

`GET /api/health` 响应示例：

```json
{
  "ok": true,
  "llm": {
    "ok": true,
    "base_url": "http://192.168.3.18:8000/v1",
    "runtime": "omlx|llama-server|unknown",
    "models_ok": true,
    "stream_ok": true
  },
  "asr": { "ok": true, "mode": "docker-cpu" },
  "tts": { "ok": true },
  "version": "mvp"
}
```

### 8.4 预检命令

README 需要提供以下预检：

```bash
curl -fsS http://192.168.3.18:8000/v1/models
curl -fsS http://<mac-mini-ip>:<frontend-port>/
curl -fsS http://<mac-mini-ip>:<api-port>/api/health
docker compose ps
docker stats --no-stream
```

---

## 9. 内存资源规划

资源验收只统计 Mac mini 本机。`192.168.3.18` 的 LLM 资源单独观测，不进入 Mac mini 16G 预算。

| 项 | 预算 | 验收口径 |
|---|---|---|
| Docker Desktop VM | <= 4GB limit | Docker Desktop 设置页或 CLI 配置 |
| 4 容器 RSS 合计 | <= 4GB 峰值 | `docker stats --no-stream` 采样 |
| whisper-asr | 0.75-1.2GB | 加载 `ggml-medium` 后稳定值 |
| piper-tts | 0.3-0.6GB | 加载目标音色后稳定值 |
| fastapi | 0.2-0.8GB | 完整对话后峰值 |
| frontend-nginx | < 0.1GB | 稳定值 |
| macOS memory pressure | 5 分钟无 critical | Activity Monitor 或 `memory_pressure` |
| swap 增量 | <= 512MB | 完整验收流程前后对比 |

不把 Docker Desktop VM 预算和容器 RSS 简单相加作为同一指标；NFR-2 的硬指标是“4 容器 RSS 合计峰值 <= 4GB，同时 macOS memory pressure 无 critical”。

---

## 10. 项目目录结构

```text
Astra/
  frontend/
    src/
      components/
      hooks/
      pages/
      store/
      lib/
    package.json
    vite.config.ts
    tailwind.config.js

  backend/
    app/
      main.py
      api/
        http_routes.py
        ws_session.py
      core/
        session_manager.py
        pipeline.py
        interrupt.py
      models/
        llm_client.py
        asr_client.py
        tts_client.py
      schemas/
        ws.py
        config.py
    requirements.txt
    Dockerfile
    .dockerignore

  deploy/
    docker/
      whisper-asr.Dockerfile
      tts.Dockerfile
      frontend.Dockerfile
    models/
      download_whisper_model.sh
      download_piper_voice.sh
    README.md

  docker-compose.yml
  README.md
```

目录要求：

- `llm_client.py` 只实现 OpenAI 兼容客户端，不实现 `omlx` 私有协议或 `llama-server` 私有协议。
- `deploy/README.md` 写明 `192.168.3.18` 的 `omlx` / `llama-server` 启动示例和 OpenAI 兼容验收命令。
- 实现代码、部署文件和 README 中不得出现旧方案残留：`Qwen3-ASR`、`Ollama in Docker`、`ASR_BITS`、`/api/generate`、自定义 `/cancel`。

---

## 11. 关键风险与缓解

| 风险 | 严重度 | 缓解 |
|---|---|---|
| `192.168.3.18` 的 `omlx` 或 `llama-server` OpenAI 兼容程度不一致 | 高 | FastAPI 只认统一 smoke test；不兼容则在远端加适配层，Mac mini 不接私有协议 |
| 远端 LLM 无法证明 request abort 后立即停止推理 | 高 | 文档拆分本地真停和远端 best-effort cancel；要求远端日志或指标作为增强验收 |
| Docker Desktop on Mac 无 GPU 透传 | 高 | 容器内 ASR/TTS 默认 CPU；Metal 只走 host 原生备选路径 |
| ASR 整段转写延迟偏高 | 中 | MVP 先接受句末转写；后续再引入 partial 或更小模型 |
| Piper HTTP server 契约随社区镜像变化 | 中 | 固定镜像 tag/digest；实现阶段写合约测试 |
| 16G 内存触发 swap | 中 | 4 容器 RSS <= 4GB，Docker VM limit <= 4GB，验收 memory pressure |
| VAD 误触发 | 中 | RMS 阈值 + 持续时间；TTS 刚开始 500ms 内可屏蔽自激；后续换 Silero VAD |

---

## 12. 后续迁移路径

| 模块 | MVP | 迁移目标 | 策略 |
|---|---|---|---|
| 后端网关 | Python FastAPI | Go 网关 | 保持 WebSocket 和 HTTP API 协议不变 |
| LLM | `192.168.3.18` OpenAI 兼容服务 | 同协议替换模型或运行时 | 只改 `LLM_BASE_URL`、`LLM_MODEL` 或远端启动方式 |
| ASR | whisper.cpp Docker CPU | whisper.cpp host Metal | 只改 `ASR_ENDPOINT` |
| TTS | Piper HTTP | 更高质量 TTS | 新增 `BaseTTSClient` 子类 |
| 前端 | React Web | Tauri / Capacitor | 复用业务 UI 和 WebSocket 协议 |
| 翻译助手 | 无 | 新增独立 pipeline | 复用 ASR/TTS，增加翻译模型服务 |

---

## 13. MVP 验收标准

### AC-1 部署可用

- Mac mini 上 `docker compose up -d` 后，4 个容器进入 healthy。
- `GET /api/health` 返回 `ok=true`，且 `llm.ok=true`、`asr.ok=true`、`tts.ok=true`。
- `192.168.3.18` 的 `/v1/models` 和流式 chat smoke test 通过。

### AC-2 核心闭环

- 局域网 Chrome 浏览器打开 Web 页面。
- 点击开始通话，说一句中文，例如“今天北京天气怎么样？”
- 前端 VAD 静音 800ms 后触发 `speech_end`。
- 后端返回 `asr_final`，语义正确率达到可用基线。
- 前端逐 token 展示 LLM 输出。
- Piper 合成至少一句语音并播放清晰。

### AC-3 VAD 自动打断

- 助手播放长句且剩余至少 2 秒时，用户插话 300ms 以上。
- 前端 300ms 内静音。
- 后端 500ms 内取消旧 `generation_id` 的 LLM/TTS 任务。
- 前端状态切回 `LISTENING`。
- 旧 `generation_id` 的 token、`tts_chunk`、`tts_end` 不再影响 UI。
- 2s 内 `whisper-asr` / `piper-tts` CPU 回落到 idle + 5% 以内。
- 远端 LLM 记录 client disconnected / request aborted；若无日志或指标，记录为 best-effort cancel。

### AC-4 手动停止

- 任意状态点击停止按钮。
- 500ms 内录音关闭、播放停止、旧任务取消。
- 状态切回 `IDLE`。
- 旧 `generation_id` 不再产生 UI 更新或音频播放。

### AC-5 延迟

10 次典型短句采样：

- 首 token P95 <= 4s，计算口径为 `speech_end` 到首个 `llm_token`。
- 首声 P95 <= 6s，计算口径为 `speech_end` 到首个可听 TTS 音频。
- 若瓶颈来自 `192.168.3.18`，验收报告必须标注远端模型、运行时和实测首 token 延迟。

### AC-6 多轮上下文

连续提问：

1. “我叫小明。”
2. “我叫什么名字？”

第二轮助手回答必须包含“小明”。

### AC-7 配置生效

- 修改 `temperature=0.1` 和短 system prompt 后保存。
- `GET /api/config` 返回 API key 掩码。
- `PUT /api/config` 请求体与服务端生效值一致。
- 后续 LLM 请求使用新配置。

### AC-8 旧方案清理

以下 grep 必须为 0：

```bash
rg 'Qwen3-ASR|ASR_BITS|Ollama.*Docker|/api/generate|/cancel' backend frontend deploy docker-compose.yml README.md
```

允许在设计文档的背景、风险和验收命令中出现这些关键字；实现代码、部署文件和 README 不得出现。

### AC-9 资源占用

- 完成 3 次完整对话，其中至少 1 次包含 VAD 打断。
- `docker stats --no-stream` 采样显示 4 容器 RSS 合计峰值 <= 4GB。
- macOS memory pressure 5 分钟无 critical。
- swap 增量 <= 512MB。

---

## 14. 外部核对来源

实现阶段需要在 `deploy/README.md` 固化以下来源或等价官方/项目链接：

- `llama-server` OpenAI 兼容接口说明。
- 当前选定 `omlx` OpenAI 兼容 server 启动说明。
- whisper.cpp server `/inference` HTTP 契约。
- Piper HTTP server `/synthesize` 契约。
- Docker Desktop on Mac 资源限制说明。
