# Astra 语音助手 MVP - The Implementation Plan (Decomposed and Prioritized Task List)

## [ ] Task 1: 基础设施与部署脚手架（目录结构 + Docker Compose 骨架）
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 创建 Monorepo 目录：`frontend/`、`backend/`、`deploy/`、根 `docker-compose.yml`、`.gitignore`
  - `deploy/docker/` 下 3 个 Dockerfile 骨架：`whisper-asr.Dockerfile`、`tts.Dockerfile`、`frontend.Dockerfile`（ollama 移除，LLM 复用已有 192.168.3.18）
  - `deploy/models/` 下 2 个下载脚本骨架：`download_whisper_model.sh`、`download_piper_voice.sh`（download_ollama_model.sh 移除）
  - 根 `docker-compose.yml` 定义 4 个本地服务（whisper-asr / piper-tts / fastapi / frontend-nginx）的 services、网络、volumes、healthcheck 骨架、端口映射、depends_on 依赖；fastapi 容器通过环境变量注入 `LLM_ENDPOINT=http://192.168.3.18/v1`、`LLM_API_KEY=`（可选）、`ASR_ENDPOINT=http://whisper-asr:PORT`、`TTS_ENDPOINT=http://piper-tts:PORT`，全部支持 env 覆盖（如 ASR 选 host 原生模式时，`ASR_ENDPOINT=http://host.docker.internal:PORT` 指向宿主机）
  - `deploy/README.md` 记录部署步骤与前置条件，包含 192.168.3.18 LLM 服务网络可达验证
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: `docker compose config` 校验通过无报错，4 个本地服务名称正确（whisper-asr/piper-tts/fastapi/frontend-nginx），LLM_ENDPOINT/ASR_ENDPOINT/TTS_ENDPOINT 环境变量注入 fastapi 容器；网络与 volume 声明存在
  - `programmatic` TR-1.2: 目录树符合 spec 第 8 章定义（frontend/src、backend/app、deploy/{docker,models}）
  - `human-judgement` TR-1.3: 代码审查 `.gitignore` 覆盖 `.superpowers/`、`node_modules/`、`deploy/models/**/*`、`__pycache__/`、`.env` 等
- **Notes**: 本任务只做骨架，不做真正的模型服务启动配置，那是 Task 5 的事；FastAPI 容器需要 `network_mode: host` 或者通过 `extra_hosts` 把 host 网关加进来以保证能访问 192.168.3.18（按 Docker Desktop 实际情况选）

## [ ] Task 2: 后端模型适配层（llm_client / asr_client / tts_client 抽象 + 空实现）
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 创建 `backend/app/models/` 目录及 3 个模块：
    - `llm_client.py`: 抽象接口 `BaseLLMClient`（`async chat_stream(messages, *, model, temperature, top_p, system_prompt, max_tokens, cancel_token) -> AsyncGenerator[str, None]`），**持有关闭 httpx stream 的句柄，暴露 `async close()` 方法供 interrupt 调用**；Mock 实现 `MockLLMClient`（返回固定流式文字）；MVP 默认实现 `OpenAICompatLLMClient` 占位（调用 httpx 发 POST 到 192.168.3.18/v1/chat/completions，stream=true SSE 解析，**cancel 只用 httpx stream `aclose()` + asyncio task cancel，绝不调用任何非标准 `/cancel` 端点**）；预留 `LlamaServerLLMClient` 接口签名（未来本地备选）
    - `asr_client.py`: 抽象接口 `BaseASRClient`（`async transcribe(wav_bytes: bytes, cancel_token) -> str`，MVP 只做整段转写，返回 final 文本；若 whisper.cpp server 支持 partial 则后续扩展方法，不阻塞 MVP），Mock 实现 + `WhisperCppAsrClient` 占位（调 whisper.cpp `/inference` HTTP）
    - `tts_client.py`: 抽象接口 `BaseTTSClient`（`async synthesize(text: str, *, voice: str, cancel_token) -> bytes`，MVP 按句子级一进一出，返回完整 wav bytes；取消 = 关闭 httpx 请求），Mock 实现（生成静音或固定波形）+ `PiperHttpTtsClient` 占位（调 Piper HTTP `/synthesize`，契约见 spec Constraints）
  - Pydantic schemas 定义：`schemas/llm.py`（OpenAI 兼容请求/响应 schema）、`schemas/asr.py`、`schemas/tts.py`、`schemas/session.py`
  - `requirements.txt` 固定 FastAPI、uvicorn、pydantic、websockets、httpx[http2]、python-dotenv、sse-starlette（或 httpx SSE 解析）等基础依赖版本
- **Acceptance Criteria Addressed**: AC-8
- **Test Requirements**:
  - `programmatic` TR-2.1: 存在 `BaseLLMClient.chat_stream`、`BaseASRClient.transcribe`、`BaseTTSClient.synthesize` 三个抽象方法签名，参数与返回类型严格匹配（llm 返回 AsyncGenerator[str]、asr 返回 str、tts 返回 bytes）；LLM 客户端对象具备可调用的 `async close()` 方法
  - `programmatic` TR-2.2: 写单元测试验证 `MockLLMClient/MockASRClient/MockTTSClient` 可正确返回数据，cancel_token 触发时立即停止（50ms 内抛 CancelledError）；LLM Mock 调用 `close()` 后 generator 停止
  - `human-judgement` TR-2.3: 代码审查 `backend/app/core/pipeline.py`（占位）中仅通过接口 import（`from ..models.llm_client import BaseLLMClient`），不直接依赖 `OpenAICompatLLMClient` 或 `WhisperCppAsrClient` 或 `PiperHttpTtsClient` 具体类
- **Notes**: Mock 客户端用于后续 Task 3/4 在没有真实模型的情况下跑通端到端闭环；OpenAICompatLLMClient 的 SSE 解析要兼容标准 `data: {...}` 格式和 `data: [DONE]` 结束标记

## [ ] Task 3: 后端核心 - 会话状态机、打断逻辑、ASR→LLM→TTS 流式流水线
- **Priority**: high
- **Depends On**: Task 2
- **Description**:
  - `core/session_manager.py`: 会话生命周期（创建/销毁/TTL 清理），状态枚举 `idle/listening/reasoning/speaking` 与转移校验（非法转移抛异常），按会话 ID 维护最近 10 轮消息历史
  - `core/interrupt.py`: 取消令牌（基于 `asyncio.Event` + `asyncio.Task` cancel），`InterruptableTaskGroup` 对 TaskGroup 封装统一 cancel 接口
  - `core/pipeline.py`: `VoicePipeline` 把 ASR→LLM→TTS 串起来；句子粒度切割 LLM 输出（凑够一句就触发 TTS）；所有步骤支持 cancel
  - `core/vad.py`（可选服务端兜底）：简单能量阈值 VAD，校验连续静音时长
  - `app/schemas/` 补充 WebSocket 消息 Pydantic 模型
- **Acceptance Criteria Addressed**: AC-3, AC-4, AC-6
- **Test Requirements**:
  - `programmatic` TR-3.1: 状态机单元测试：覆盖 12+ 种合法转移 + 3+ 种非法转移（如从 idle 直接跳到 speaking）应被拒绝
  - `programmatic` TR-3.2: 打断测试：用 Mock 客户端跑 pipeline，在 TTS 流式 yield 过程中调用 cancel，断言 generator 在 50ms 内停止抛出 `CancelledError`
  - `programmatic` TR-3.3: 上下文记忆：构造 12 轮对话送入 session，断言只保留最近 10 轮（最早 2 轮被裁剪）
  - `human-judgement` TR-3.4: 代码审查 `InterruptableTaskGroup` 的 cancel 路径是否正确关闭所有子任务（无遗漏）
- **Notes**: interrupt 路径必须覆盖所有 pending 的 httpx 请求，建议统一用一个 httpx.AsyncClient 并传 timeout

## [ ] Task 4: 后端路由层 - HTTP 配置 API + WebSocket 语音会话（打通 Mock 适配层）
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - `app/main.py`: FastAPI 应用入口，CORS 中间件（允许局域网 `*` 用于 MVP），挂载路由，启动时注入默认配置（LLM_ENDPOINT 从 env 读取 `http://192.168.3.18/v1`）
  - `app/api/http_routes.py`:
    - `GET /api/health` 健康检查（检查 LLM(192.168.3.18) + ASR + TTS 共 3 个服务连通性；LLM 通过 `GET /v1/models` 或一次简短 `POST /v1/chat/completions` 探活）
    - `GET /api/config` 获取当前运行配置（LLM endpoint+api_key 掩码返回 / model / temperature / top_p / system_prompt + ASR/TTS 端点 + VAD 参数）
    - `PUT /api/config` 更新运行时配置（非持久化 MVP；api_key 更新后立刻生效到 LLM 客户端）
  - `app/api/ws_session.py`:
    - `WS /ws/session`：接收 `start_session`、二进制 `audio_chunk`、`interrupt`、`end_session`
    - 发送 `state_change`、`asr_partial`、`asr_final`、`llm_token`、`tts_chunk`、`tts_end`、`error`
    - 心跳 30s ping/pong；断线自动清理会话
    - 集成 Task 2 的 Mock 客户端，先跑通无真实模型的闭环
  - `backend/Dockerfile`：基于 python:3.11-slim，装依赖 + uvicorn 启动，健康检查 `GET /api/health`
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-7, AC-9
- **Test Requirements**:
  - `programmatic` TR-4.1: `GET /api/health` 返回 200，JSON 含 keys `["llm_ok", "asr_ok", "tts_ok"]`（mock 下均 true；联调后真实 llm_ok 要求 192.168.3.18 返回 2xx）
  - `programmatic` TR-4.2: `PUT /api/config` 修改 temperature / system_prompt / llm_endpoint 后，`GET /api/config` 返回一致；api_key 在响应中被掩码（只显示首尾 2 字符）
  - `programmatic` TR-4.3: 用 `websockets` 库写集成测试：连 `/ws/session` → 发 start_session → 注入一段假的 `audio_chunk`（通过 Mock ASR 已内置识别）→ 断言收到 `asr_final` + 多条 `llm_token` + 多条 `tts_chunk` + `tts_end`
  - `programmatic` TR-4.4: 在上述 tts_chunk 流中发 `interrupt`，断言 500ms 内收到最后一条消息且不再有新分片
- **Notes**: 先通 Mock 客户端，真实模型客户端在 Task 5 中补；api_key 只能存后端，绝不写入前端 JS bundle 或 localStorage

## [ ] Task 5: 真实模型服务客户端接入（OpenAI 兼容 LLM 192.168.3.18 + whisper.cpp ASR + Piper TTS HTTP）
- **Priority**: high
- **Depends On**: Task 4
- **Description**:
  - `llm_client.py`: 完成 `OpenAICompatLLMClient`，参数化 endpoint（默认 `http://192.168.3.18/v1`）/ api_key / model；**唯一端点** `POST {endpoint}/chat/completions` 带 `stream=true`，SSE 解析逐 token（标准 `data: {...}` JSON 中取 `choices[0].delta.content`，遇到 `data: [DONE]` 结束）；**支持取消只用两步**：① `await httpx_stream_response.aclose()` 关闭 SSE HTTP 连接 ② 上层 `asyncio.Task.cancel()`；**绝不调用任何非标准 `/cancel` 或 `/api/generate` 端点**（grep backend/app 下这两个字符串必须 0 行）；从环境变量 `LLM_ENDPOINT` / `LLM_API_KEY` 或 `/api/config` 读 model / temperature / top_p / system_prompt；鉴权头 `Authorization: Bearer {api_key}`（仅当 api_key 非空时添加）；**客户端实例内部持有关闭流所需的 httpx.Response 句柄，并通过 `async close()` 对外暴露**
  - `asr_client.py`: 完成 `WhisperCppAsrClient`，**调 whisper.cpp 自带 server 的 HTTP `/inference` 接口**（非自研，避免 Docker 内装 qwen-asr/transformers/bitsandbytes）；构造 POST multipart/form-data，字段 `file` 传整段 wav bytes、`temperature`/`language` 可选；返回 final 文本字符串；环境变量 `ASR_ENDPOINT` 默认指向 `http://whisper-asr:8080`（Docker 内），允许用户改到 host 原生 Metal 模式（如 `http://host.docker.internal:8080`）；**MVP 不实现 partial streaming，只返回 final，若后续需 partial 可加 `/stream` 接口扩展，不影响当前抽象**
  - `tts_client.py`: 完成 `PiperHttpTtsClient`，**收敛到社区 HTTP server 镜像固定契约**（见 spec Constraints）：POST `{TTS_ENDPOINT}/synthesize` body `{ "text": str, "voice": "zh_CN-huayan-medium"|"en_US-lessac-medium" }`，接收 `audio/wav` bytes，固定采样率 22050Hz、单声道 16-bit；**取消 = 关闭 httpx 请求**，不依赖 Piper 自定义 `/cancel` endpoint；MVP 严格按句子级一进一出（凑够一句调一次，返回完整 wav bytes 给前端播放队列，不做 token 级 chunked）
  - `deploy/docker/whisper-asr.Dockerfile`：**基于 `ggerganov/whisper.cpp:latest-server` 官方镜像（或等效 CPU NEON 优化镜像）**，启动时 entrypoint 加载 ggml-medium.bin，暴露 `/inference` + `/health`；不要装 python/transformers/bitsandbytes 任何依赖（彻底移除 Qwen3-ASR 相关所有链路）
  - `deploy/docker/tts.Dockerfile`：**基于 `docker.io/rhasspy/piper-http-server:latest`（或等效兼容社区镜像）**，挂载中英文音色文件，暴露 `/synthesize` + `/health`；契约严格对齐 spec：入参 JSON {text, voice}、出参 audio/wav 22050Hz mono 16-bit
  - 2 个 `download_*.sh` 脚本补全：① `download_whisper_model.sh` 从官方 ggml 仓库下载 `ggml-medium.bin`（约 750MB）到 `/models/whisper/`；② `download_piper_voice.sh` 从官方 ModelScope/HF 下载 `zh_CN-huayan-medium` + `en_US-lessac-medium` 两个音色（合计约 300MB）到 `/models/piper/`；全部放到 volume 挂载目录
  - 192.168.3.18 网络连通性验证：在 fastapi 容器内写一个探活脚本（healthcheck 的一部分）：`curl -sf http://192.168.3.18/v1/models`（若服务需鉴权则带 Bearer token 调）；`GET /api/health` 的 `llm_ok` 字段基于此探活结果
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-3, AC-4, AC-5, AC-7, AC-8
- **Test Requirements**:
  - `programmatic` TR-5.1: `curl -sf http://192.168.3.18/v1/models`（或带 Bearer token）返回 2xx；10 次采样往返延迟 P95 ≤ 200ms；用 OpenAICompatLLMClient 发一次简短 10 字符流式请求，5s 内返回至少 1 个 token；**cancel 时先 `await client.close()` 再 `task.cancel()`，断言 httpx Response `is_closed == True` 且打断后 2 秒内不再收到新 token**
  - `programmatic` TR-5.2: 用 `httpx` 调 whisper.cpp 容器 `/inference` 接口，喂一段 5s 标准中文 wav（16kHz 单声道 16-bit），断言在 6s 内拿到非空 final 结果文字（字符数 ≥ 5，语义 ≥ 80% 字正确率）；**不要求 partial**
  - `programmatic` TR-5.3: 调 Piper TTS 容器 `/synthesize` 接口输入一句中文（≥10 字，voice=zh_CN-huayan-medium），断言返回 Content-Type 为 `audio/wav`，wav 字节流 ≥ 20KB，时长 > 0.5 秒；**用 soundfile/python-wave 解析验证：采样率=22050、通道数=1、位深度=16**
  - `human-judgement` TR-5.4: Mac mini 本地 4 个容器 + Docker runtime 实际内存占用（`docker stats --no-stream` 汇总 whisper-asr/piper-tts/fastapi/frontend-nginx 四个容器 RSS 之和），**全服务峰值 ≤ 4GB**（持续运行 5 分钟采样，含 3 次完整对话）；`memory_pressure` 无 critical 事件；swap usage 增长 ≤ 512MB（不频繁）
  - `programmatic` TR-5.5: 从 Task 4.3 的集成测试基础上替换 Mock → 真实客户端，端到端跑同样的断言（asr_final 非空、llm_token 至少 1 条、tts_chunk 至少 1 条、tts_end）
  - `programmatic` TR-5.6: **打断真停验证（落实 AC-3/AC-4）**：启动 3 客户端跑一次长对话（LLM 至少要输出 5 秒以上未完成），在 SPEAKING 播放中途发 interrupt，断言：① httpx stream connection 确实关闭（`is_closed==True`）；② `docker stats` 采样 whisper-asr 与 piper-tts 的 CPU%，打断前 busy、打断后 2s 内回落至 idle±5%；③ 保持安静 3s，听不见 TTS 漏播尾巴
  - `programmatic` TR-5.7: **grep 禁用端点（落实 NFR-6 / AC-8-3）**：在 `backend/app/` 全局执行 `rg -n "/api/generate|/cancel"`，结果行数 = 0
- **Notes**: OpenAI 兼容接口的 `stop` 参数、`max_tokens` 参数、`top_p` 默认值要与 LLM 服务实际支持情况对齐，必要时在客户端做适配层过滤；Apple Silicon Docker 访问宿主机局域网若遇问题，优先尝试 `extra_hosts: {"host-gateway": host-gateway}` 或 `network_mode: "host"`（Mac 版 Docker Desktop host-mode 有限制，需实测选一种可用方案）；若 rhasspy/piper-http-server 镜像 `/synthesize` 契约与 spec 不一致，在适配层做归一化（不要改 Constraints 契约）

## [ ] Task 6: 前端 React 骨架（路由、状态、基础 UI）
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - Vite + React 18 + TS 初始化项目，安装 zustand、react-router-dom、tailwindcss、shadcn/ui 基础依赖
  - `src/store/session.ts`: Zustand store 维护 `state`（idle/listening/reasoning/speaking）、`messages[]`、`config`、`wsStatus`、`currentDuration`
  - `src/pages/CallPage.tsx`: 通话主页面布局（顶部状态栏 + 中部对话气泡滚动区 + 底部控制区）
  - `src/pages/ConfigPage.tsx`: 配置页面（表单：LLM 模型名、temperature、top_p、system prompt、LLM OpenAI 兼容 endpoint（默认 `http://192.168.3.18/v1`）、api_key（password 类型输入，前端只传不存 localStorage）、ASR/TTS 服务 URL、VAD 阈值）
  - `src/components/ui/`: shadcn 基础组件（Button、Card、Input、Slider、Switch、Dialog、Toaster）
  - `deploy/docker/frontend.Dockerfile`: 两阶段构建（build vite → 产物 copy 到 nginx:alpine），conf.d 默认支持 SPA history fallback + `/api` 反向代理到 fastapi 容器
- **Acceptance Criteria Addressed**: AC-8（可移植性：无平台特有 API）
- **Test Requirements**:
  - `programmatic` TR-6.1: `npm run build` 成功无 TS 报错
  - `programmatic` TR-6.2: Vitest 单测 session store：dispatch 状态变更（如 set_state("listening")）后 getState() 断言 state === "listening"
  - `programmatic` TR-6.3: 构建前端 Docker 镜像后启动容器，`curl localhost:80` 返回包含 React root div 的 HTML
  - `human-judgement` TR-6.4: 代码审查：整个 frontend/src 不出现 `window.__TAURI__`、`Capacitor`、Electron 等平台专有 API（保证后续套壳不改动）
- **Notes**: 先做骨架，不做录音/WebSocket/波形，Task 7/8 做

## [ ] Task 7: 前端核心 Hooks（useWebSocket、useRecorder、useVAD、useAudioPlayer）
- **Priority**: high
- **Depends On**: Task 6
- **Description**:
  - `useWebSocket(url)`: 原生 WebSocket 封装，30s 心跳、指数退避重连、发送 JSON/二进制区分、暴露 `sendJson/sendBytes/onMessage/readyState`；断线事件推到 store
  - `useRecorder()`: 基于 MediaRecorder + AudioContext 做 16kHz 16bit PCM 重采样（若 Safari 不支持 MediaRecorder 原生 PCM 就用 ScriptProcessor/AudioWorklet）；暴露 start/stop/pause、`onChunk(bytes)` 回调、权限请求；输出 20ms 一包（640 bytes）
  - `useVAD(rmsThreshold=0.01, speechDurationMs=300, silenceDurationMs=800)`: 对麦克风输入每 10ms 取一次 RMS；连续超阈值 300ms → 发 onSpeechStart；连续低于阈值 800ms → onSpeechEnd；暴露手动灵敏度调节
  - `useAudioPlayer(sampleRate=22050)`: 基于 AudioContext 自建播放队列，appendChunk(bytes) 流式入队，支持 flush() 立即打断清空；onStart / onEnd 回调；TTS 分片按 wav 头解析或固定 PCM 格式
- **Acceptance Criteria Addressed**: AC-2, AC-3, AC-4, AC-9
- **Test Requirements**:
  - `programmatic` TR-7.1: 用 jsdom + vitest fake timers 测 useWebSocket：断开后 3s 触发首次重连，最多重连 5 次后停止
  - `human-judgement` TR-7.2: 真实 Chrome 浏览器手动点"允许麦克风"后 useRecorder 正常回调 onChunk，ArrayBuffer 长度稳定 640（20ms 16kHz 16bit）
  - `programmatic` TR-7.3: 给 useVAD 喂合成的 RMS 序列（350ms 高 + 1000ms 低），断言 onSpeechStart / onSpeechEnd 各触发 1 次，时间戳差在允许误差
  - `human-judgement` TR-7.4: useAudioPlayer 接入一段 wav bytes，手动 flush() 时播放立刻静音，onEnd 回调在 ≤ 300ms 内触发
- **Notes**: 这些 hooks 高度依赖浏览器 API，单元测试以 mock 为主，辅以 checklist 里的手动验证

## [ ] Task 8: 前端业务组件与页面集成（波形、气泡、控制按钮、打断交互）
- **Priority**: high
- **Depends On**: Task 7, Task 4（后端 Mock 通了即可对调）
- **Description**:
  - `components/Waveform.tsx`: Canvas 绘制实时波形，useRecorder 实时 PCM 数据驱动，LISTENING 态绿色动、SPEAKING 态粉色动、其他态灰色
  - `components/ChatBubble.tsx`: 区分 user/assistant 气泡；user 气泡在 asr_partial 时显示灰色草稿，asr_final 后变黑；assistant 气泡随 llm_token 逐字追加，左边有状态小圆点（reasoning 转圈/speaking 喇叭）
  - `components/ControlBar.tsx`: 底部控制区，含「开始通话」大按钮 / 「打断停止」红按钮互斥显示 + 当前状态文字 + 时长计时
  - `CallPage.tsx` 集成：按状态驱动录音开关/VAD 监听；VAD onSpeechStart 在 SPEAKING 态立即发 interrupt；VAD onSpeechEnd 在 LISTENING 态触发让后端开始推理（或后端自判，二选一，但前端必须同步通知后端一个可选的 speech_end 消息）
  - 配置页与主页面路由联通（右上角齿轮按钮进入 ConfigPage），保存后写 store 并 PUT `/api/config`
- **Acceptance Criteria Addressed**: AC-2, AC-3, AC-4, AC-6, AC-7
- **Test Requirements**:
  - `programmatic` TR-8.1: Playwright E2E：打开页面 → mock 麦克风权限 → 点「开始通话」 → 断言 store state 变为 listening，Waveform canvas 有绘制（getContext 调用数>0）
  - `human-judgement` TR-8.2: 手动打断流程：在「说话中」播放状态，拍一下手（或制造 300ms 噪音）→ 应立即停止播放并切回 LISTENING；或点红按钮 → 立即切回 IDLE
  - `programmatic` TR-8.3: E2E 中修改配置页 temperature 为 0.1 并保存，断言前端 fetch 发出了 `PUT /api/config` 请求体包含 temperature: 0.1
- **Notes**: E2E 测试要给浏览器 mock 麦克风与音频输出设备，避免真实录音

## [ ] Task 9: 端到端联调、Docker Compose 全流程打通、文档补齐
- **Priority**: high
- **Depends On**: Task 5, Task 8
- **Description**:
  - 把前端 env 的 `VITE_WS_URL` / `VITE_API_URL` 指向 Docker 内部网络（通过 nginx 的 /api 和 /ws 反代即可避免跨域）
  - `docker-compose.yml` 各容器 healthcheck 补全（whisper-asr `/health`、piper-tts `/health`、fastapi `/api/health` 含 192.168.3.18 探活、frontend-nginx 200）
  - 实测 4 个本地服务同时启动后内存与 swap，如超限则微调（无 ASR_BITS 概念，whisper.cpp medium 模型大小固定）；验证 fastapi 容器确实能访问 192.168.3.18（host-gateway / extra_hosts / host-mode 三选一，按 Docker Desktop 实测结果）
  - 写根 `README.md`：架构简图、硬件要求、Docker 前置、LLM 192.168.3.18 可访问性预检、3 步部署流程（1. 下载 whisper/TTS 模型 2. compose up 3. 浏览器打开）、ASR 切换 Docker/host 原生 Metal 两种方式、常见问题排查
  - `deploy/README.md` 细化：每个 Docker 镜像是怎么构建的、**环境变量清单**（LLM_ENDPOINT / LLM_API_KEY / ASR_ENDPOINT / TTS_ENDPOINT / VITE_API_URL / VITE_WS_URL）、如何把 ASR_ENDPOINT 改到 `http://host.docker.internal:8080` 以启用 host 原生 Metal、如何切换 LLM endpoint / model / api_key、Docker 访问局域网 LLM 时的网络配置选项、**真停指标的手动观测方式**（docker stats CPU% 回落 / 后端 debug 日志 token 流停止）
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-3, AC-4, AC-5, AC-7, AC-9
- **Test Requirements**:
  - `programmatic` TR-9.1: 从零开始（空 models 卷、空镜像缓存，192.168.3.18 已就绪），按 README 步骤执行脚本，断言 4 个本地容器（whisper-asr / piper-tts / fastapi / frontend-nginx）3 分钟内全部 healthy 且 `GET /api/health` 返回 `{llm_ok: true, asr_ok: true, tts_ok: true}`
  - `human-judgement` TR-9.2: **完整端到端验收走查**（对应 AC-2 闭环 + AC-3 VAD 打断 + AC-4 红按钮打断 + AC-6 上下文记忆 + AC-7 配置生效）至少重复 3 次，中间插重启 fastapi 容器验证断线重连（AC-9）；**每次走查 VAD 打断后保持安静 3s，确认听不见 TTS 漏播尾巴**
  - `human-judgement` TR-9.3: **延迟指标埋点 10 次采样（落实 AC-5）**：典型短句 5~8 秒，从「前端 VAD onSpeechEnd + generation_id 创建」打 T0 戳，记录 T1(首条 llm_token) 与 T2(useAudioPlayer onStart)；最终结果：P95(T1) ≤ 4s、P95(T2) ≤ 6s；若 192.168.3.18 本身 RTT 慢导致超标，必须在文档中标注原因
  - `human-judgement` TR-9.4: **资源占用验收（落实 NFR-2）**：连续 3 次完整对话（含打断）后，`docker stats --no-stream` 汇总 4 容器内存峰值合计 ≤ 4GB；`memory_pressure` 5 分钟峰值不触发 critical；swap usage 增长 ≤ 512MB（不频繁）
  - `human-judgement` TR-9.5: **真停验收（落实 NFR-4）**：录制 1 次打断全过程的时间戳 + `docker stats` CPU% 采样，要求：① 前端 300ms 内静音；② 后端 500ms 内关闭 SSE 连接 + cancel task；③ whisper-asr/piper-tts CPU% 打断后 2s 内回落至 idle±5%；④ 不影响下一轮 generation（不误杀）
- **Notes**: 本任务为集成收口，允许小幅回改 Task 1-8 的任何问题；192.168.3.18 若存在 SSE 非标准格式（如一次性返回完整 JSON），允许在 OpenAICompatLLMClient 里补兼容分支但接口签名不变；若 rhasspy/piper-http-server 镜像不可用，允许换 fork 镜像但适配层契约不变（POST /synthesize、返回 22050Hz wav）
