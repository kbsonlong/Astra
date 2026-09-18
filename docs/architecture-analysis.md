# Astra 当前架构

> 更新：2026-09-19
> 定位：在 Apple Silicon Mac mini 上运行的局域网会议语音工作台。

## 运行边界

- FastAPI、MLX/音频模型和持久化状态原生运行在 Mac mini；Docker Compose 只承载前端 Nginx。
- 远端 LLM 只通过 OpenAI 兼容 HTTP 接口接入；Astra 不依赖任何提供方私有运行时协议。
- 真实录音、模型权重、声纹库和会议产物均在本机文件系统或 SQLite 中保存，不提交到 Git。

## 后端分层

```text
浏览器 / Nginx
  ├─ HTTP: 转写、会议、审核、声纹、训练、配置
  └─ WebSocket: 实时会话、会议状态、通知
                 │
FastAPI API 层 ──┼── 鉴权、上传限制、IP 并发限制、任务状态
                 │
核心编排层      ├── 实时 VoicePipeline: ASR -> LLM -> TTS
                 └── 会议 MeetingPipeline:
                     VAD -> ASR -> 标点 -> 说话人/分离 -> 确认规则纠错
                     -> 纪要 -> 翻译 -> 审核数据导出
                 │
模型适配层      ├── MLX ASR/TTS、FunASR、Piper、OpenAI 兼容 LLM
                 └── 可选增强、回声消除、分离与重叠检测
```

实时 ASR 使用固定 worker 串行处理，以避免 MLX 同步推理阻塞多个请求；会议和训练改由独立子进程运行，避免长任务占用 API 事件循环。

## 可靠性与资源边界

- `TaskStore` 以 SQLite 登记会议和训练进程，应用启动时收敛遗留任务；任务具备并发上限、超时、取消和终态产物清理。
- 上传在后端按字节与可解码时长限制，会议大文件分块写盘；实时请求、上传与 WebSocket 共享按 IP 的并发额度。
- 会议处理失败时，增强、分离或标点等可选阶段会记录降级原因并保留可用 ASR 结果。
- 实时 WebSocket 的 `generation_id` 只在单个连接内有效。浏览器重连时重置本地代际水位，防止新 Session 的事件被误判为旧事件。
- 浏览器只保留实时 PCM 帧缓冲；麦克风 PCM 会分块追加到 Mac mini 的 `REALTIME_RECORDING_DIR`，停止时原子封装为 WAV。单次录音默认上限 2 GiB，已完成与未完成文件默认保留 7 天。

## 前端结构

- React Router 管理实时助手、工作台、审核、训练和设置路由，导航无需整页重载。
- `live/session.ts` 保存可单测的重连代际和录音上限规则；`live/VoiceDial.tsx` 独立承载实时状态可视化。
- 实时会话仍由 `App.tsx` 协调 MediaStream、AudioContext、PCM 帧与 TTS 排队；后续新增协议或设备类型时，应继续抽成专门 hook，而不是扩张页面组件。

## 安全模型

配置 `ADMIN_TOKEN` 后，除认证端点外的 HTTP 与 WebSocket 均要求管理员会话。会话 Cookie 使用 `HttpOnly` 与 `SameSite=Strict`；部署 HTTPS 时必须启用 `AUTH_COOKIE_SECURE=true`。未配置令牌的开放模式只兼容可信局域网。

## 验证层级

CI 执行后端 Ruff、受限范围的 Mypy、pytest，以及前端 Vitest 与生产构建。这证明类型、接口和纯逻辑路径；不证明麦克风、浏览器后台恢复、MLX 内存、模型质量、远端 LLM 或长音频 SLA。真实环境验收见 [acceptance-checklist.md](acceptance-checklist.md)。
