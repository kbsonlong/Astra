# Astra 语音助手 MVP - Product Requirement Document

## Overview
- **Summary**: 基于局域网环境（Mac mini 16G Apple Silicon + 已有 LLM 服务）构建的完全本地化实时语音通话助手。MVP 阶段通过 React Web 页面提供服务，采用三层架构：客户端（React Web）→ 后端网关（Python FastAPI，后续迁移到 Go）→ 模型服务层。模型服务层部署策略：① LLM 复用已有 OpenAI 兼容接口 192.168.3.18；② ASR 用 whisper.cpp（Apple Silicon 可验证路径，支持 host 原生或 Docker CPU，默认 Docker CPU）；③ TTS 用 Piper 封装好的 HTTP server（社区维护镜像）。核心交互为「前端 VAD 端点检测 → 整段 ASR → LLM `/v1/chat/completions` SSE 流式 → 按句子级 TTS 合成 → 播放」，支持 VAD 自动打断与手动按钮打断双模式。ASR 和 TTS 完全本地处理，LLM 请求在局域网内完成，不离开内网。
- **Purpose**: 在消费级硬件（16G Mac mini）上验证一套端到端低延迟、隐私可控、可落地性强的本地语音助手可行性，复用已有的 LLM 推理服务，为后续扩展到桌面端 / 移动端和翻译助手打基础。
- **Target Users**: 初期为开发者和本地 AI 爱好者，在同一局域网内通过浏览器访问；后续扩展为日常办公个人用户。

## Goals
- 在 16G Mac mini 上部署架构分层清晰：**Docker Compose 编排 FastAPI + 前端 Nginx + Piper TTS + whisper.cpp ASR 共 4 个容器**（LLM 复用已有 192.168.3.18 OpenAI 兼容接口，不本地部署；Docker Desktop on Mac 不做 GPU/MPS 假设，所有容器内推理均走 CPU 或使用 whisper.cpp 针对 Apple 优化的 CPU NEON/BLAS 路径；未来若需 Metal 加速再走 host 原生部署）
- 跑通「用户说话 → 前端 VAD 静音 800ms 端点 → 整段 wav 发送 ASR → LLM 流式回答 → 按句子切分送 TTS → 语音播放」完整端到端闭环
- 实现流畅的打断体验（VAD 自动 + 手动按钮双模式）：打断后前端播放立即静音、后端 LLM HTTP SSE 连接关闭 + asyncio task cancel、TTS 合成取消、ASR 未提交的 buffer 按 generation_id 丢弃
- LLM 首 token 延迟（从 ASR final 提交到首条 llm_token）P95 ≤ 4s；TTS 首包播放延迟（从 ASR final 提交到前端喇叭出声）P95 ≤ 6s
- 至少保持最近 10 轮对话上下文记忆
- MVP 采用 FastAPI + OpenAI 兼容 LLM 客户端，同时为后续 Go 网关 + llama-server 本地备选预留清晰适配层
- 后续跨平台（macOS/Linux Tauri 桌面端、iOS/Android 移动端）不重构核心 Web 代码
- **ASR partial streaming 作为迭代增强项（非 MVP blocker）：MVP 验收允许只有整段转写（asr_final 一条），partial 若 whisper.cpp server 支持则顺带开启，不支持不阻塞 MVP 交付**

## Non-Goals (Out of Scope)
- 翻译助手模块（MVP 之后独立迭代）
- 会话持久化（数据库/文件保存历史）和历史会话回溯
- 多模型切换、模型热更新、多用户鉴权、多租户
- 特定语音指令识别与动作执行
- Tauri 桌面端封装、移动端（Capacitor/PWA）打包（MVP 仅浏览器访问）
- 任何调用外部云 API 的行为（Redis、MySQL 等第三方组件必须使用 Docker，不直接安装）
- **Docker 内 GPU/MPS/ANE 透传加速（Docker Desktop on Mac 不支持，不作为 MVP 目标）；若未来要走 Metal，模型服务改到 host 原生进程部署即可，接口不变**
- **ASR partial streaming（字级流式识别）；MVP 以端点后整段转写为必做基线，partial 仅做可选项**
- **token 级 TTS 流式；MVP 严格按句子级流水线：凑够一句 → 合成一句 → 播放一句**

## Background & Context
- 硬件底座：用户已有 Mac mini 16G（Apple Silicon），需要部署本地 ASR + TTS + 后端网关 + 前端静态服务；LLM 推理服务已独立部署于局域网机器 `192.168.3.18`，提供 OpenAI 兼容接口（`/v1/chat/completions` stream=true SSE）。
- **Docker Desktop on Mac GPU 限制**：评审确认 Docker Desktop on Mac 不支持 GPU/MPS/ANE 透传，因此所有 Docker 内的本地推理统一走 whisper.cpp CPU NEON/BLAS 优化或 Piper CPU 路径；如果后续需要 Metal 加速（ASR/LLM），改为在宿主机跑独立进程即可，适配层接口不变。
- **ASR 选型原则**：Qwen3-ASR-0.6B 流式仅支持 vLLM/CUDA 场景，在 Apple Silicon Docker CPU 路径上可验证性不足；MVP 切换为 whisper.cpp（ggml 模型 + CPU NEON 加速 + 自带 `/inference` HTTP server + 整段转写稳定），内存占用 < 1GB（medium 模型约 750MB），在 16G 机型上安全。
- **TTS 选型原则**：Piper 原生只有 CLI/库，MVP 收敛到社区维护的具体 HTTP server 镜像和固定契约（见 Constraints），不自己造 server。
- **打断语义原则**：不依赖 Ollama/OpenAI 兼容服务的「取消 endpoint」；后端统一通过持有 httpx stream connection + asyncio task，在 interrupt 时先关闭连接再 cancel task；同时记录打断前后 192.168.3.18/本地服务的 CPU/GPU 指标验证真正停止。
- **模型优先级**：效果优先，接受 2~4s 延迟；LLM 复用 192.168.3.18；ASR whisper.cpp medium（中文+英文够用）；TTS Piper 中文+英文音色。
- 架构倾向：前后端分离，Monorepo 管理；后端网关通过统一适配层调用各模型服务，便于 FastAPI → Go 切换和 LLM/ASR/TTS 多后端切换。
- 跨平台路线：React Web MVP → Tauri 桌面端（同一套 React 代码封装）→ 移动端（Capacitor / PWA），核心 WebSocket 交互逻辑保持不变。
- 第三方组件一律 Docker 化，不直接安装到 Mac mini 宿主机；**例外是未来若确需 Metal 加速的模型服务，允许在 host 原生进程部署，纳入「host 原生服务」清单管理即可**。

## Functional Requirements
- **FR-1**: 用户在 Web 页面点击"开始通话"后，能在 Mac mini 服务端建立独立会话（含 session_id + generation_id 用于取消追踪），前端持续采集麦克风音频（16kHz PCM 16bit 单声道）通过 WebSocket 分片上传到后端录音 buffer
- **FR-2**: **前端**做 VAD 端点检测（连续静音 > 800ms 判定句末），句末时向前端内部 pipeline 提交「整段 wav + generation_id」，自动触发 ASR → LLM → TTS 流水线；**后端不再做 VAD，只保留兜底超时断句**
- **FR-3**: ASR 通过 whisper.cpp server（ggml medium 模型，CPU NEON/BLAS，Docker 部署可选 host 原生）对 FR-2 提交的整段 wav 执行转写，**MVP 至少返回 1 条 `asr_final`**；若服务端支持 partial（whisper.cpp 某些分支有 timestamp-based partial），可选推送 `asr_partial`，但不作为 MVP 验收项
- **FR-4**: LLM 基于最近 10 轮对话上下文 + ASR final 文本构造 messages，调用 192.168.3.18 OpenAI 兼容 `POST /v1/chat/completions`（**只用 `/chat/completions`，绝不调 `/api/generate`**），SSE stream=true 逐 token 返回（llm_token），后端持有关闭流所需的 httpx 响应对象
- **FR-5**: TTS 接收 LLM **句子级增量**文本（以中文/英文标点切句，凑够一句就发一次合成，不等待整段结束），通过 Piper HTTP server 统一契约合成音频，按固定采样率的 wav/PCM chunk 推送前端（tts_chunk）；支持随时取消当前合成
- **FR-6**: SPEAKING 播放阶段通过前端 VAD 检测用户插话（连续 300ms 超音量阈值），立即触发 interrupt：① 前端 useAudioPlayer.flush() 静音；② 前端 WS 发 `interrupt(generation_id=X)`；③ 后端拿到 generation_id → cancel 对应 LLM asyncio task + 关闭 httpx SSE 连接 → cancel TTS task → 丢弃尚未提交的 ASR buffer；④ 状态切回 LISTENING
- **FR-7**: UI 提供"打断/停止"按钮，任意状态下点击立即执行 FR-6 的中断链路（区别是最后一步状态切到 IDLE 而非 LISTENING）
- **FR-8**: UI 实时展示状态指示（idle/listening/reasoning/speaking）、录音波形、对话历史气泡（用户/助手角色）、当前通话计时
- **FR-9**: 提供配置页面，可修改 LLM 模型名、temperature、top_p、system prompt、OpenAI 兼容 endpoint（含 api_key，默认值 `http://192.168.3.18/v1`）、ASR/TTS 服务端点 URL、VAD 灵敏度；api_key 仅在后端保存，GET /api/config 返回时做掩码
- **FR-10**: 提供 Docker Compose 一键启动 **whisper.cpp ASR、Piper TTS HTTP、FastAPI、前端静态 Nginx 共 4 个容器**；提供模型下载脚本（whisper.cpp ggml medium 模型、Piper 中英文音色）；**LLM 复用 192.168.3.18，不本地部署，不强依赖 Metal in Docker**；允许「ASR 改 host 原生部署」的替代方案，compose 通过 env `ASR_ENDPOINT` 切换目标地址
- **FR-11**: 后端和各模型服务之间通过适配层接口解耦：LLM 客户端、ASR 客户端、TTS 客户端独立模块，后续实现替换时不影响上层会话流水线；**每个客户端构造函数接收 endpoint 参数，支持通过配置热切换**
- **FR-12**: WebSocket 具备心跳（30s ping）与断线自动重连，后端会话具备 TTL 僵尸清理机制；**所有运行中任务按 (session_id, generation_id) 索引，cancel 按 key 精确命中，不误杀其他轮次**

## Non-Functional Requirements
- **NFR-1（性能）**: 在 Mac mini 16G + 局域网 192.168.3.18 LLM 服务环境下，典型短句对话（≤10秒录音）从「VAD 判定句末 + generation_id 创建」计时：**LLM 首 token 到达前端 P95 ≤ 4s**；**TTS 首包播放出声 P95 ≤ 6s**；LLM 往返与 ASR 转写耗时全部计入（不单独豁免）
- **NFR-2（资源占用）**: Mac mini 本地 4 个容器（asr(whisper.cpp medium) / tts / fastapi / frontend）常驻内存峰值合计 ≤ **4GB**（其中 whisper.cpp medium ~750MB、Piper ~300MB、FastAPI ~800MB、Nginx 可忽略，Docker runtime 预留 2GB），保证 macOS + Docker 正常运行不触发频繁 swap；LLM 服务 192.168.3.18 资源不计入
- **NFR-3（隐私合规）**: 音频采集、ASR、TTS 在 Mac mini 本地处理完成；LLM 请求在局域网内（Mac mini → 192.168.3.18）完成，绝不外发公网外部请求；api_key 仅保存在 Mac mini 后端内存，不写入前端 bundle、localStorage、日志（debug 日志也要脱敏）
- **NFR-4（可靠性 - 打断真停）**: 单次 interrupt 下发后：① 前端 300ms 内静音；② 后端在 500ms 内完成 **httpx stream aclose() + asyncio.Task cancel()** 两步；③ **打断后 2s 内 192.168.3.18/本地 ASR/TTS 的 CPU/GPU 指标回落（若可观测）**；绝不出现"前端停了后端还在跑模型吃算力"
- **NFR-5（可移植性）**: 前端 React 代码不引入平台特有 API，同一套代码可无缝迁移到 Tauri 壳 / Capacitor 壳；核心 WebSocket 消息协议在迁移过程中保持不变
- **NFR-6（可维护性）**: 模型适配层（llm_client / asr_client / tts_client）对外提供统一抽象接口；**LLM 客户端只用 OpenAI `/chat/completions` 一个端点，不允许再引入 `/api/generate`、`/cancel` 等 Ollama 特定端点**；后续 LLM 本地新增 llama-server 备选时新增子类即可，上层 pipeline 0 改动

## Constraints
- **Technical**:
  - 前端：React 18 + TypeScript + Vite + Zustand + shadcn/ui + Tailwind CSS
  - MVP 后端：Python 3.11 + FastAPI（异步 WebSocket + asyncio + httpx AsyncClient）
  - 长期后端：Go（Gin/Fiber，消息协议不变时替换）
  - **MVP LLM 服务（唯一端点）**: 局域网 `http://192.168.3.18/v1/chat/completions`（SSE stream=true，标准 OpenAI 请求体，只用这一个端点）
  - LLM 本地备选（后续可选）：llama-server（llama.cpp server）本地部署，走 host 原生（不是 Docker in-container GPU）
  - **ASR 服务**: whisper.cpp server（ggml medium 模型，CPU NEON/BLAS 路径；默认 Docker 部署，`ASR_ENDPOINT` 允许指向 host 原生进程以启用 Metal）；**绝不使用 Qwen3-ASR + vLLM/transformers in Docker（Apple 无 CUDA，流式无法验证）**
  - **TTS 服务（收敛契约）**: Piper HTTP server，社区维护镜像（优先 `docker.io/rhasspy/piper-http-server:latest` 或等效兼容镜像）；**固定契约**：
    - 输入：`POST /synthesize` body `{ "text": str, "voice": "zh_CN-huayan-medium"|"en_US-lessac-medium" }`
    - 输出：`audio/wav` bytes，固定采样率 22050Hz，单声道 16-bit；HTTP 响应可 chunked transfer；MVP 不要求 token 级 chunked，句子级一进一出即可
    - 中断：取消 httpx 请求即终止当前合成；后端不依赖 Piper 特定 `/cancel` endpoint
  - 部署：Docker + Docker Compose，4 个本地容器；模型文件通过 Volume 挂载；LLM_ENDPOINT / LLM_API_KEY / ASR_ENDPOINT / TTS_ENDPOINT 全部支持 env 覆盖
  - 第三方组件：Redis/MySQL 等如果用到，必须 Docker 部署，禁止直接安装宿主机；**允许 host 原生部署 whisper.cpp / llama-server 以便利用 Metal，必须在文档中标注「host 原生服务清单」**
- **Business**:
  - 硬件上限固定 Mac mini 16G
  - MVP 单用户场景，不做多并发优化
  - MVP 「端到端可用」优先级高于「字级流式体验」，因此 ASR/TTS 流式只做可选项而非必做项
- **Dependencies**:
  - **whisper.cpp**（ggml medium 模型 + 自带 HTTP server / 或 `ggerganov/whisper.cpp` 官方 server 镜像）— Apple Silicon CPU 可验证路径
  - 192.168.3.18 OpenAI 兼容 `POST /v1/chat/completions` SSE 流式接口 — LLM 唯一入口
  - httpx (Python) AsyncClient，用于：① LLM SSE 调用（可 `.aclose()` 取消）② TTS `/synthesize`（可 `.cancel()`）③ ASR `/inference`
  - Piper TTS HTTP server 社区镜像（`rhasspy/piper-http-server` 或兼容 fork）
  - React 浏览器端 MediaRecorder + Web Audio API（麦克风权限 + 16kHz PCM 重采样）+ 前端 VAD（RMS 阈值 + 时间窗）

## Assumptions
- Mac mini、访问浏览器、LLM 服务 192.168.3.18 三者在同一局域网，互 ping 延迟 ≤ 20ms；跨公网访问不在 MVP 范围
- Mac mini 已安装 Docker Desktop（或 Docker Engine + Compose）并正常运行；Docker Desktop on Mac 无 GPU 透传，所有容器内模型走 CPU 路径（whisper.cpp 已针对 Apple CPU NEON 优化）
- 浏览器端允许麦克风权限（Chrome/Safari）；HTTPS 环境或 localhost 环境下 MediaRecorder API 可正常工作
- 192.168.3.18 支持 SSE 标准格式：`data: {...}\n\n` + 结束帧 `data: [DONE]\n\n`；支持标准参数 `model / messages / temperature / top_p / max_tokens / stop / stream`；若有差异在适配层归一化
- 192.168.3.18 鉴权方式三种之一：① 无鉴权；② `Authorization: Bearer <api_key>`；③ 自定义 header（在配置页加 `extra_headers` 配置，暂不实现复杂签名）
- whisper.cpp Docker/host 模式可任选，在文档中给出两种启动方式；medium 模型中文识别能达到日常对话 80%+ 字正确率
- Piper HTTP server 的 `/synthesize` 响应可以是单次 200 wav bytes（MVP 够用），若以后升级 chunked 再增强适配层；句子级一进一出足够
- 模型文件（whisper.cpp ggml medium / Piper 中英文音色）首次启动时可联网从官方/HF/ModelScope 下载，或用户可提前手动放到挂载卷

## Acceptance Criteria

### AC-1: Docker Compose + Host 原生服务组合部署成功
- **Given**: Mac mini 16G 已安装 Docker Desktop/Engine，代码已 clone，模型下载脚本（whisper.cpp ggml medium + Piper 音色）已执行完成；LLM 服务 192.168.3.18 网络可达（`curl -sf http://192.168.3.18/v1/models` 返回 2xx）
- **When**: 执行 `docker compose up -d`（默认模式：whisper.cpp 在 Docker CPU 路径；备选模式 compose 停掉 whisper.cpp，改 host 原生启动，通过 env `ASR_ENDPOINT=http://host.docker.internal:PORT` 指向）
- **Then**: 3 分钟内 4 个本地容器（**whisper-asr**、piper-tts、fastapi、frontend-nginx）全部 healthy；前端页面 `http://<mac-mini-ip>:<port>` 可访问；后端 `GET /api/health` 返回：`{ "llm_ok": true, "asr_ok": true, "tts_ok": true }`（三服务都通）
- **Verification**: `programmatic`
- **Notes**: `docker ps --format "table {{.Names}}\t{{.Status}}"` 验证容器状态；ASR 如果选 host 原生模式，whisper-asr 容器可以 unhealthy/不存在，但 `asr_ok` 仍需 true（证明 host 原生地址可达）

### AC-2: 端到端语音对话闭环跑通（ASR 整段转写基线 + LLM 流式 + TTS 句子级）
- **Given**: 浏览器打开页面、已授权麦克风、4 个容器 healthy、192.168.3.18 连通性正常
- **When**: 用户点击"开始通话" → 说一句中文（如"今天北京天气怎么样？"）→ 停止说话 → 前端 VAD 检测到 800ms 静音并触发句末
- **Then**: ① 用户气泡出现一条 asr_final 文字，内容语义正确（≥ 80% 字正确率）；**partial 不要求，不作为失败条件**；② 助手气泡逐 token 流式出现 LLM 回答文字，来源是 192.168.3.18 `/v1/chat/completions` SSE；③ 文字到标点处触发 TTS 句子级合成，语音自动播放，内容与文字对应一致；④ 整个过程按 generation_id 递增编号，可追踪
- **Verification**: `programmatic`（后端日志抓包断言 asr_final 非空 + 至少 10 条 llm_token + 至少 1 条 tts_chunk）+ `human-judgment`（人工听语义通顺、识别内容和语音内容一致性）

### AC-3: VAD 自动打断生效（真停）
- **Given**: 助手正在 SPEAKING 播放 TTS 长句回复（至少还剩 2 秒未播完）；LLM 推理也可能还在进行
- **When**: 用户持续说话 ≥ 300ms 触发前端 VAD onSpeechStart（模拟插话）
- **Then**:
  1. 前端 ≤ 300ms 内静音（audio player flush）
  2. 后端收到 `interrupt(generation_id=X)` 后 ≤ 500ms 完成两步动作：**a) `await httpx_stream_response.aclose()` 关闭 LLM SSE 连接 b) `llm_task.cancel()` + `tts_task.cancel()`**
  3. 状态机立刻切到 LISTENING，UI 状态指示同步更新
  4. **打断后 2 秒内 192.168.3.18 若可观测 CPU/GPU（或通过下一次 `/health` 采样对比），以及本地 ASR/TTS 容器 `docker stats` CPU%，应回落至空闲水平（即真的停了，不是前端停了后端还在烧算力）**
  5. 继续监听麦克风，下一次 VAD onSpeechEnd 能正常进入新一轮 ASR（即不误杀其他 generation）
- **Verification**: `programmatic`（时间戳断言 + httpx `is_closed` 检查 + `docker stats` 采样 CPU% 打断前后对比）+ `human-judgment`（打断后立刻保持安静 3s，听是否有漏出来的 TTS 尾巴）

### AC-4: 手动按钮打断生效（真停 + 状态到 IDLE）
- **Given**: 任意状态（LISTENING 录音中 / REASONING 等 LLM / SPEAKING 播放中）；至少有一个后台任务（推理 / 合成 / 录音）在跑
- **When**: 用户点击红色"打断/停止"按钮
- **Then**: 500ms 内所有 4 项全部终止：① useRecorder.stop() 录音关闭；② useAudioPlayer.flush() 静音；③ LLM SSE 连接 aclose + task cancel；④ ASR/TTS 未完成合成取消；状态切到 **IDLE**；打断后 2 秒内 `docker stats` 采样的 ASR/TTS CPU% 与 192.168.3.18 `/health` 侧 CPU 指标（若可获取）回落到空闲区间
- **Verification**: `programmatic`（状态位 + cpu%回落 + cancel 捕获）

### AC-5: 体感延迟指标达标（从句末计时）
- **Given**: 典型短句场景，用户说 5~8 秒中文问句，普通话清晰，环境噪音 ≤ 50dB
- **When**: 从「前端 VAD 触发 onSpeechEnd + generation_id 创建」那一刻打时间戳 T0，连续采样 ≥ 10 轮对话
- **Then**:
  1. **T1（首字）**：前端收到首条 `llm_token` 的 P95 ≤ 4s（含 ASR 转写耗时 + LLM RTT）
  2. **T2（首声）**：前端 `useAudioPlayer` 回调 onStart 即真正喇叭出声的 P95 ≤ 6s（含 TTS 首句合成耗时 + 首包传输）
- **Verification**: `programmatic`（后端埋点日志，10 次采样计算 P95；异常重试不超过 3 次）
- **Notes**: 如果 192.168.3.18 单点延迟高导致 T1 超标，需在文档中注明「LLM 服务本身 RTT 是瓶颈，非本系统流水线问题」；否则必须达标

### AC-6: 多轮上下文记忆
- **Given**: 一轮对话正常进行中，历史上下文最近 10 轮已开启；整个过程无打断
- **When**: 第一轮用户说"我叫小明，记住这个名字" → asr_final 识别正确 → LLM 回复确认 → 第二轮用户问"我叫什么名字？" → asr_final 正确
- **Then**: 第二轮 LLM 输出文本（`llm_token` 拼接起来的完整字符串）明确包含"小明"字样（大小写不敏感，可容忍标点错误）
- **Verification**: `programmatic`（断言 完整 assistant output contains "小明"）

### AC-7: 配置项修改生效（收敛到 `/chat/completions`）
- **Given**: 用户进入配置页面；当前使用默认 192.168.3.18 `/v1/chat/completions`
- **When**: 修改：`temperature = 0.1`、`top_p = 0.9`、`system_prompt = "你是一个只会回复两到四个字的助手，绝不多说。"`、（可选）`model = 新值` / `api_key = 新值` → 保存 → 发起 3 轮独立新对话（不要上下文串联）
- **Then**:
  1. `PUT /api/config` 成功，`GET /api/config` 返回的 temperature/top_p/system_prompt/model/llm_endpoint 与修改一致；**api_key 被掩码（只显首尾各 2 字符或省略）**
  2. 后端发送到 192.168.3.18 `/v1/chat/completions` 的请求体里，`temperature === 0.1`、`top_p === 0.9`、`messages[0].role === "system"` 且 content 为新的 system prompt；**请求路径严格只有 `/v1/chat/completions`，绝不出现 `/api/generate` 或任何 `/cancel` 端点**
  3. 主观可见：LLM 回答明显更短（≤ 10 字为主）、重复问句多次回答相似度更高（temperature=0.1 的低随机性）
- **Verification**: `human-judgment`（3 次重复问句观察一致性与长度）+ `programmatic`（拦截 LLM 客户端请求体断言参数与路径）

### AC-8: 适配层接口隔离 & 只有 `/chat/completions` 单一 LLM 入口
- **Given**: 后端代码存在，模型适配层抽成独立模块
- **When**: 审查 `backend/app/models/` 下 llm_client.py / asr_client.py / tts_client.py 与上层 `pipeline.py`、`ws_session.py`
- **Then**:
  1. 每个模块对外暴露统一抽象类：`BaseLLMClient.chat_stream()` / `BaseASRClient.transcribe()`（MVP 可只做整段，不用 stream 后缀，避免和之前的 Qwen partial stream 混淆）/ `BaseTTSClient.synthesize()`
  2. 上层 `pipeline.py`、`ws_session.py` 所有 import 都只引抽象类，绝不直接 `import OpenAICompatLLMClient`、`WhisperCppAsrClient`、`PiperHttpTtsClient`
  3. **在 `backend/app/` 下全局 grep `/api/generate`、`/cancel` 结果为 0 行**；LLM 客户端所有 HTTP 调用只有 `POST {llm_endpoint}/chat/completions`
  4. 新增 `LlamaServerLLMClient`（host 原生 llama.cpp server）作为代码里的 stub/NotImplemented 占位，只需实现抽象的空方法，证明上层不用改；无需真正跑通
- **Verification**: `human-judgment`（代码审查）+ `programmatic`（grep 断言禁用端点 0 命中）

### AC-9: WebSocket 心跳 & 断线重连 & 会话僵尸清理
- **Given**: 页面建立 WebSocket 连接，30s ping/pong 心跳正常；会话存在 (session_id=S1)
- **When**: 手动 `docker restart astra-fastapi-1`（重启 fastapi 容器模拟断线）
- **Then**:
  1. 前端 10 秒内检测断线并自动重连成功；UI 弹出 toast「已重连」，或可见 wsStatus 从 disconnected → connected
  2. 新的 session_id=S2 创建成功，S1 不再接收新消息
  3. 10 分钟后 S1 会话在后端 session_manager 内存中被清理（会话计数不再增长；可通过 debug endpoint / prometheus 指标观测 session_count 回落 1）
- **Verification**: `programmatic`（模拟断线 + ws 状态断言 + 10 分钟后 session 数量断言）

## Open Questions
- [ ] 192.168.3.18 的鉴权方式：① 无鉴权 ② `Authorization: Bearer <api_key>` ③ 自定义 header（如 X-API-Key）—— 需确认后写入配置页默认值与适配层
- [ ] 192.168.3.18 的 SSE 格式：严格 OpenAI 标准（`data: {...}\n\n` + `data: [DONE]\n\n`）还是有自定义扩展？如果有差异需在 `OpenAICompatLLMClient` 补归一化分支
- [ ] whisper.cpp 选择哪种部署方式作为 MVP 默认：① Docker CPU（开箱即用，推荐 MVP 默认）② host 原生 server + Metal 加速（延迟/吞吐更好，但需用户装依赖）—— 确认后写进 README 默认步骤
- [ ] Piper HTTP server 具体镜像选型：`docker.io/rhasspy/piper-http-server` 社区维护版是否可用？如果其 `/synthesize` 入参/出参与契约（POST JSON、返回 wav 22050Hz mono 16-bit）不一致，MVP 是换镜像还是在适配层归一化？
- [ ] 浏览器端 Safari 对 MediaRecorder API 的 PCM 16kHz 原生支持度不足时，是否需要引入 AudioWorklet 做重采样兜底（MVP 先只支持 Chrome，还是必须兼容 Safari？）
- [ ] 打断后 192.168.3.18 的「CPU/GPU 是否真停」如何观测：是提供额外的 LLM 服务监控 endpoint（用户侧），还是只验证本端 httpx connection 已关闭 + 日志不再收到 token？如果 LLM 服务没有监控 endpoint，MVP 可以只验「本端关闭+无更多 token」作为真停判据（需用户确认可接受）
