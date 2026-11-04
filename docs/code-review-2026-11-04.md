# Code Review — 2026-11-04（未推送的 12 个提交）

评审范围：`origin/main..HEAD` 共 12 个提交（实时 TTS 二进制帧、`/api/config`
allowlist、SQLite WAL、profile 拆分、会议纪要引用/断点续跑/并行 fan-out、numpy
2.x 迁移）。工作区在评审时无未提交改动。

本文件先列出发现（按严重程度与文件/行），再给出修复说明与验证。

## 发现

### 高
- 正常会议流程丢弃待办/决策结果。`backend/app/core/meeting.py` 非续跑路径调用
  `self.summarize(...)` 时未透传 `on_note_task` 与 `note_tasks=result.note_tasks`。
  fan-out 仍会发起 LLM 请求, 但结果只写进 `MeetingContext` 内部新建的 dict,
  `result.note_tasks` 恒为空, 导致 `report.md` 无「待办/关键决策」章节、
  `status.json` 无子任务进度。现有测试直接调用 `summarize()`, 未覆盖 `process()`。

### 中
- 摘要可能因超时被清空。`meeting.py` 的 `wait_for` 对摘要也套用
  `MEETING_NOTE_TASK_TIMEOUT_SECONDS`(默认 300s)。长会议分块串行易超时,
  超时后摘要为空, 连带翻译/引用统计为空, 主产物丢失。
- 三路并发可能撑爆 16GB 机型内存。`_run_note_tasks` 用 `gather` 同时发起
  三路 ~6000 token 的 LLM 请求, 与「分块以规避 KV 内存上限」的初衷冲突。
- `realtime` profile 下 `/transcribe/stream` 静默降级。该路由从
  `app.state.meeting_pipeline` 取 VAD/增强/分离, 而 realtime 不构建会议管线,
  结果三者为 None, 接口不报错但退化为整文件 ASR, 转写质量下降。
- 拆分部署未按路由分流。`frontend/vite.config.ts` 把 `/api` 与 `/ws` 都指向
  8001(workbench), 而 workbench 不挂 `/ws`; Nginx 默认 `API_UPSTREAM` 指向
  8000(realtime), 会议/训练接口 404。`deploy/README.md` 写了需分流但无对应配置。
- 续跑任务审校/训练数据为空。checkpoint 不存音频, 续跑后 `Segment.audio=None`,
  `_write_training_artifacts` 跳过每段, 新任务的 `transcript_segments.jsonl`
  与 `asr_clips/` 为空。
- `validate_indextts_l6.py` 失效。脚本仍要求 `tts_chunk`/`audio_b64`, 已被
  二进制 ATTS 帧取代。

### 低
- 同时携带 `file` 与 `resume_of` 的请求必然失败(新目录无 `segments.json`)。
- `resume_of` 不校验源任务状态, 可续跑运行中/成功的任务。
- realtime 仍挂载音色库写接口, 与 README「只读」描述不符。
- `test_app_profiles.py` 写死 `cwd="backend"`, 在 `backend/` 下跑 pytest 报错。
- `tts_voice_store.py` 的 `enable_wal` import 插在标准库 import 中间。
- timeline 的 `seg-XXXXXX` 与训练导出(合并段)的 `seg-XXXXXX` 同格式不同域,
  易混淆。

## 修复

已修复（本次提交）：
- 高：`meeting.py` 非续跑路径的 `summarize(...)` 补传 `on_note_task` 与
  `note_tasks=result.note_tasks`；新增 `test_process_non_resume_populates_note_tasks`
  走 `process()`(桩掉 decode/transcribe) 作回归守卫。
- 中(摘要超时)：新增 `MEETING_SUMMARY_TIMEOUT_SECONDS`(默认 1800s), 摘要与翻译
  用更宽松超时, 行式待办/决策仍用 `MEETING_NOTE_TASK_TIMEOUT_SECONDS`(300s)。
- 中(并发内存)：新增 `MEETING_NOTE_TASK_MAX_CONCURRENCY`(默认 3, 保留历史并行;
  16GB 机型设 1 即串行), `_run_note_tasks` 用信号量限制真正在途的 LLM 请求数。
- 中(stream 降级)：realtime-only 进程额外构建仅含 VAD/增强/分离的轻量
  `MeetingPipeline`(无会议 LLM/ASR fan-out), `/transcribe/stream` 保持分块。
- 中(代理分流)：`vite.config.(ts|js)` 支持 `VITE_API_UPSTREAM`/`VITE_WS_UPSTREAM`,
  默认指向单进程 `all` 的 8000, 拆分时可分别指向 realtime/workbench。
- 中(validate 脚本)：`validate_indextts_l6.py` 改用底层 `receive()` 分流,
  二进制帧经 `decode_tts_frame` 解析, 必需事件改为 `tts_audio_frame`。
- 低：`/api/meeting/process` 拒绝 `file`+`resume_of` 同给(400); 仅允许续跑
  `failed`/`done` 任务, 其余返回 409；新增两条路由守卫测试。
- 低：修正 `tts_voice_store.py` 的 import 顺序。
- 低：`test_app_profiles.py` 以文件路径解析 backend 根目录替代写死 `cwd`。
- 低：新增三个设置纳入 `/api/config` allowlist 与 `.env.example`。
- 低(文档)：`deploy/README.md` 更正 realtime 的音色库/stream 说明。

未改（已评估为变更风险大于收益，记为已知项）：
- 续跑任务的审校/训练数据为空：需从源任务复制/引用音频产物或 UI 明示,
  属功能性增强, 留作后续。
- `seg-XXXXXX` 双域命名：训练 ID 已落盘于 `transcript_segments.jsonl` 并被
  `_SEGMENT_ID_RE` 与审校/音频 URL 契约消费, 重命名为跨切面破坏性改动,
  当前保留澄清注释。
- realtime 音色库写接口拆分为只读子集：需拆 router 且牵涉前端用法,
  本次以文档对齐 + 建议反代层放行只读路径, 代码拆分留作后续。

## 验证
- 后端：`PYTHONPATH=backend .venv/bin/python -m pytest backend/tests -q`。
- 前端：`npx vitest run`。
