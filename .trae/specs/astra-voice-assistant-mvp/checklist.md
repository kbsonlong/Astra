# Astra 语音助手 MVP - 验证清单（Verification Checklist）

## Task 1: 基础设施与部署脚手架
- [ ] Checkpoint 1.1: `docker compose config` 无报错，识别到 4 个本地 service（whisper-asr / piper-tts / fastapi / frontend-nginx）；fastapi 容器 environment 中出现 `LLM_ENDPOINT=http://192.168.3.18/v1`、`ASR_ENDPOINT=...`、`TTS_ENDPOINT=...` 注入
- [ ] Checkpoint 1.2: 目录结构符合 spec：frontend/src、backend/app/api、backend/app/core、backend/app/models、backend/app/schemas、deploy/docker、deploy/models 均存在；deploy/docker 下有 whisper-asr.Dockerfile / tts.Dockerfile / frontend.Dockerfile 共 3 个（ollama.Dockerfile 不存在）；deploy/models 下有 download_whisper_model.sh / download_piper_voice.sh 共 2 个
- [ ] Checkpoint 1.3: 根 `.gitignore` 包含 `node_modules/`、`__pycache__/`、`.superpowers/`、`deploy/models/*`（保留占位 .gitkeep）、`.env`、`*.pyc`、`.DS_Store`
- [ ] Checkpoint 1.4: deploy/README.md 写明了 Docker 前置条件、ASR/TTS 下载脚本使用说明、端口映射、**192.168.3.18 可访问性预检步骤**（`curl -sf http://192.168.3.18/v1/models`）、**ASR 两种部署方式说明**（Docker CPU 默认 / host 原生 Metal 备选通过 `ASR_ENDPOINT` 切换）
- [ ] Checkpoint 1.5: 2 个 `download_*.sh` 脚本均包含 `set -euo pipefail` 和使用说明，能独立执行（虽不要求模型真的存在）；ollama 相关脚本已不存在

## Task 2: 后端模型适配层（抽象+Mock）
- [ ] Checkpoint 2.1: `backend/app/models/llm_client.py` 中存在 `BaseLLMClient` 抽象类，定义了 `async chat_stream(messages, *, model, temperature, top_p, system_prompt, cancel_token, ...) -> AsyncGenerator[str, None]` 方法签名；包含 `MockLLMClient` 和 `OpenAICompatLLMClient` 两个具体实现类；**预留 `LlamaServerLLMClient` 签名（可只写 NotImplemented 占位）**；**LLM 客户端实例具备 `async close()` 方法用于关闭 httpx SSE 流**
- [ ] Checkpoint 2.2: `backend/app/models/asr_client.py` 中 `BaseASRClient.transcribe(wav_bytes, cancel_token) -> str` 方法签名正确（MVP 整段转写，返回 final 字符串）；`WhisperCppAsrClient` 具体实现类存在占位
- [ ] Checkpoint 2.3: `backend/app/models/tts_client.py` 中 `BaseTTSClient.synthesize(text, *, voice, cancel_token) -> bytes` 方法签名正确（MVP 句子级一进一出，返回完整 wav bytes）；`PiperHttpTtsClient` 具体实现类存在占位
- [ ] Checkpoint 2.4: MockLLM / MockASR / MockTTS 三个实现的单元测试通过：可迭代/返回正确、cancel_token 触发时立即停止不出垃圾数据（50ms 内）；MockLLM 调 close() 后 generator 立刻停
- [ ] Checkpoint 2.5: 上层占位模块（如 pipeline.py 的 import）只依赖抽象接口，**`from models.llm_client import BaseLLMClient`**；不出现 `from models.llm_client import OpenAICompatLLMClient` 或 `OllamaLLMClient` / `WhisperCppAsrClient` / `PiperHttpTtsClient` 这类硬依赖
- [ ] Checkpoint 2.6: requirements.txt 中包含 `httpx[http2]` / `sse-starlette`（或同类 SSE 解析库）用于 OpenAI 兼容 SSE 调用；**不出现 `qwen-asr`、`transformers`、`bitsandbytes`、`auto-gptq` 这 4 个 Qwen3-ASR 依赖**（grep 0 行）

## Task 3: 后端核心（状态机+打断+流水线）
- [ ] Checkpoint 3.1: 状态机单元测试覆盖：`idle→listening`、`listening→reasoning`、`reasoning→speaking`、`speaking→listening`（打断）、`listening→idle`、`speaking→idle` 合法转移成功
- [ ] Checkpoint 3.2: 至少 3 种非法转移（如 `idle→speaking`、`reasoning→idle`（未经允许）、`idle→reasoning`）状态机抛出异常或被拒绝
- [ ] Checkpoint 3.3: `InterruptableTaskGroup.cancel()` 单元测试：在生成器进行中途调用，50ms 内抛出 CancelledError/自定义 InterruptException
- [ ] Checkpoint 3.4: 12 轮对话喂入 session_manager，`get_messages_for_llm()` 只返回最新 10 轮（前 2 轮被裁剪）
- [ ] Checkpoint 3.5: VoicePipeline 集成 Mock 三客户端测试：从假音频分片走到 TTS bytes 分片输出，中间发出 cancel 后 100ms 内 pipeline 结束

## Task 4: 后端路由层（HTTP+WS + Mock）
- [ ] Checkpoint 4.1: `GET /api/health` 返回 200 且包含 keys：`["llm_ok", "asr_ok", "tts_ok"]`（mock 下均为 true，llm_ok 的实现是简单探活 192.168.3.18，**mock 阶段可短路为 true**）
- [ ] Checkpoint 4.2: `PUT /api/config` 修改 `{ "temperature": 0.3, "system_prompt": "...", "llm_endpoint": "x", "api_key": "sk-xxx" }` 后，`GET /api/config` 返回值一致；**api_key 在 GET 响应中被掩码（仅显首尾字符或不返回）**
- [ ] Checkpoint 4.3: WebSocket 端到端集成测试用例通过：发 start_session → 注入假音频 → 收 state_change(listening/reasoning/speaking) + asr_partial + asr_final + ≥1 条 llm_token + ≥1 条 tts_chunk + tts_end
- [ ] Checkpoint 4.4: 中断集成测试：tts_chunk 流进行中发 interrupt，收到 state_change(listening) 后 500ms 内无更多 tts_chunk/llm_token
- [ ] Checkpoint 4.5: `backend/Dockerfile` 能正常 build，启动后健康检查接口 `GET /api/health` 通过

## Task 5: 真实模型服务接入（OpenAI 兼容 LLM 192.168.3.18 + whisper.cpp + Piper HTTP）
- [ ] Checkpoint 5.1: 192.168.3.18 连通性验证：从 fastapi 容器内 `curl -sf http://192.168.3.18/v1/models`（带 Bearer token 若需鉴权）返回 2xx；**10 次采样往返延迟 P95 ≤ 200ms**
- [ ] Checkpoint 5.2: OpenAICompatLLMClient 单元测试通过：调用 192.168.3.18 发一次最短流式请求（stream=true），3 秒内收到第一条 token；**cancel 路径只用 `await client.close()` + `asyncio.Task.cancel()`；断言 httpx Response.is_closed == True，且 2s 内不再收到新 token**；鉴权头正确（Authorization: Bearer xxx，当 api_key 非空时）；**grep 客户端源码绝不出现 `/cancel` 或 `/api/generate` 字样**
- [ ] Checkpoint 5.3: **whisper.cpp ASR 容器健康检查通过**（`/health` 200），喂一段 5s 标准中文 wav（16kHz 单声道 16-bit）能得到非空 final 文本（字符数 ≥ 5，字正确率 ≥ 80%）；**不要求 partial streaming；镜像基于 ggerganov/whisper.cpp:latest-server 或等效 CPU NEON 优化；不装 python/transformers/bitsandbytes**
- [ ] Checkpoint 5.4: **Piper TTS HTTP 容器健康检查通过**，喂一句中文（voice=zh_CN-huayan-medium）返回 Content-Type=audio/wav，wav bytes 长度 ≥ 20KB，能被 ffplay/wavesurfer 正常播放；**用 soundfile/python-wave 解析验证：采样率=22050、通道数=1、位深度=16**
- [ ] Checkpoint 5.5: 2 个 download 脚本在本地空目录执行时成功下载到 volume 挂载目录；模型文件夹总大小合理（**whisper ggml-medium ~750MB、Piper 两个音色 ~300MB，LLM 不下载**）；Qwen3-ASR 量化权重相关下载彻底不存在
- [ ] Checkpoint 5.6: **资源占用验收（落实 NFR-2）**：Mac mini `memory_pressure` 在 4 个本地容器全启动 + 连续 3 次对话后仍是非 critical 状态；`docker stats` 汇总 4 容器（whisper-asr + piper-tts + fastapi + frontend-nginx）内存 RSS 峰值 **合计 ≤ 4GB**（LLM 192.168.3.18 不计入）；swap usage 增长 ≤ 512MB（不频繁）
- [ ] Checkpoint 5.7: 用真实 192.168.3.18 LLM + whisper.cpp ASR + Piper TTS 跑通一次 Task 4.3 的端到端闭环，收到 asr_final 语义正确（≥ 80% 字正确率），TTS 播放清晰
- [ ] Checkpoint 5.8: **grep 禁用端点验证（落实 NFR-6 / AC-8-3）**：在 `backend/app/` 全局执行 `rg -n "/api/generate|/cancel"` 或等效 grep，结果行数 = 0
- [ ] Checkpoint 5.9: **打断真停验证（落实 NFR-4 / AC-3/AC-4）**：长对话中途 interrupt，断言：① httpx stream connection 确实关闭（is_closed==True）；② `docker stats` 采样 whisper-asr 与 piper-tts CPU%，打断后 2s 内回落至 idle±5%；③ 保持安静 3s 听不见 TTS 漏播尾巴

## Task 6: 前端 React 骨架
- [ ] Checkpoint 6.1: `npm run build` 零 TypeScript 错误、零 vite 构建错误
- [ ] Checkpoint 6.2: Zustand session store 单测通过：set_state、append_message、update_config 三个 action 状态变更符合预期
- [ ] Checkpoint 6.3: CallPage / ConfigPage 两个路由可通过 react-router 切换，不刷新浏览器
- [ ] Checkpoint 6.4: 前端 Dockerfile 两阶段构建成功；Nginx 启动后 `curl http://localhost:80/` 返回 HTML 包含 `<div id="root">`；访问不存在路由 `/foo` 仍返回 index.html（SPA fallback）
- [ ] Checkpoint 6.5: 代码审查 frontend/src 全局 grep `__TAURI__|Capacitor|electron|node:child_process|fs\.readFile` 等无命中，确认完全浏览器可用、无平台依赖
- [ ] Checkpoint 6.6: ConfigPage 表单包含 LLM endpoint 字段（默认 `http://192.168.3.18/v1`）和**api_key 密码输入框**；前端不将 api_key 写入 localStorage / 或日志输出
- [ ] Checkpoint 6.7: **api_key 安全验证（落实 NFR-3）**：① 配置页保存后刷新浏览器，localStorage 检查无 api_key 键；② 前端控制台（console.*）、XHR 请求体日志（Network 面板可见但非本地持久化）不打印 api_key 明文；③ PUT `/api/config` 仅通过 HTTPS/localhost 发送，不暴露到其他 JS 全局变量

## Task 7: 前端核心 Hooks（WS/录音/VAD/播放）
- [ ] Checkpoint 7.1: useWebSocket 在 fake timer 下模拟网络断开，3s 后发起首次重连（退避），第 5 次失败后停止重连并抛出 exhausted 事件
- [ ] Checkpoint 7.2: 真实 Chrome 手动验证：允许麦克风后 useRecorder.start() 成功；200ms 内收到至少 5 次 onChunk 回调；每次 bytes.length===640（20ms 16kHz 16bit）
- [ ] Checkpoint 7.3: VAD 合成输入测试：RMS 序列 400ms 超阈值 → 触发 onSpeechStart；之后 1 秒低于阈值 → 触发 onSpeechEnd；误差 ≤ ±100ms
- [ ] Checkpoint 7.4: useAudioPlayer：入队 1 秒 wav bytes，播放 0.3s 时调用 flush()，AudioContext currentTime 停止推进，onEnd 在 300ms 内触发；无内存泄漏（playback queue 已清空）

## Task 8: 前端业务组件与集成
- [ ] Checkpoint 8.1: Playwright E2E（mock 麦克风权限+mock audio）：打开 → 点开始通话 → Waveform canvas 绘制调用数 ≥ 1 帧/500ms，状态显示 LISTENING
- [ ] Checkpoint 8.2: E2E 注入真实录音 mock：对话气泡区至少 1 条 user 气泡和 1 条 assistant 气泡；assistant 气泡的文字随着时间逐步增长（流式）
- [ ] Checkpoint 8.3: 手动联调 VAD 打断：在 TTS 播放进行中拍手/说话 300ms+，播放立刻静音，状态立刻回到 LISTENING（红喇叭图标消失/绿麦克风出现）
- [ ] Checkpoint 8.4: 手动联调红按钮打断：任意状态点「停止」→ ≤ 500ms 进入 IDLE，useAudioPlayer 队列清空，useRecorder 已停止
- [ ] Checkpoint 8.5: ConfigPage 修改 temperature 和 system prompt + **修改 LLM endpoint/api_key** 保存后，Network 面板出现 `PUT /api/config` 请求体正确；下一轮对话的 LLM 风格与 temperature=0.1 一致（回答更短更确定）

## Task 9: 端到端联调 + Docker Compose + 文档
- [ ] Checkpoint 9.1: 从零开始（空 images / volumes，192.168.3.18 已就绪）按根 README 三步部署：1) 运行 2 个 download 脚本 2) `docker compose up -d` 3) 浏览器打开地址；结果：3 分钟内 4 个本地容器（whisper-asr / piper-tts / fastapi / frontend-nginx）全 healthy，**前端可访问，`/api/health` 中 llm_ok:true**
- [ ] Checkpoint 9.2: 「验收全走查」成功 3 次：
  ① 开始 → 问"今天北京天气怎么样？" → ASR 正确识别 → LLM 文字流式出现 + TTS 清晰播放（AC-2）
  ② 回复到一半手动 VAD 打断 + 红按钮打断各一次，**立刻停 + 保持安静 3s 无 TTS 尾巴**（AC-3/AC-4）
  ③ 连续 2 轮上下文问答："我叫小明" → "我叫什么？" → 答"小明"（AC-6）
- [ ] Checkpoint 9.3: **延迟埋点 10 次采样（落实 AC-5）**：VAD 结束到首条 llm_token P95 ≤ 4s；到首条 tts_chunk 播放出声 P95 ≤ 6s；若 192.168.3.18 单点慢需在文档中标注原因
- [ ] Checkpoint 9.4: **打断真停观测（落实 NFR-4）**：3 次走查中抽 1 次录制打断前后 docker stats CPU%：whisper-asr 与 piper-tts CPU% 打断后 2s 内回落至 idle±5%；后端日志确认 2s 内不再输出 llm_token 流
- [ ] Checkpoint 9.5: 重启 fastapi 容器模拟断线，前端 10s 内自动重连成功并提示「已重连」；10 分钟后旧会话被清理（内存泄漏检测：session count 不再增长）
- [ ] Checkpoint 9.6: **文档完整性审查**：根 README 有架构图/硬件要求/192.168.3.18 预检查/一键部署/ASR 两种启动方式（Docker CPU vs host 原生 Metal）/常见问题；deploy/README 写明各环境变量（**LLM_ENDPOINT / LLM_API_KEY / ASR_ENDPOINT / TTS_ENDPOINT / VITE_API_URL / VITE_WS_URL**）及如何切换 LLM endpoint/model/api_key、Docker 访问局域网方案、真停指标观测方式；**无 TBD/占位符；不出现 ASR_BITS / 4bit/8bit/Qwen3-ASR 字样**
- [ ] Checkpoint 9.7: **资源占用终验（落实 NFR-2）**：3 次完整对话（含打断）后，`docker stats` 4 容器内存 RSS 合计峰值 ≤ 4GB；`memory_pressure` 5 分钟无 critical；swap 增量 ≤ 512MB（不频繁）
- [ ] Checkpoint 9.8: 代码自查：所有 FastAPI/WS 路由有 docstring；前端 hooks 有注释；关键逻辑（打断/状态转移）每步有日志输出（debug 级别），方便后续排错
