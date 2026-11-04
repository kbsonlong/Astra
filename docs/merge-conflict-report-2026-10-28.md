# Astra 远端代码拉取与冲突处理报告

- 日期：2026-10-28
- 仓库：`/Users/zengshenglong/Code/RustWorkSpace/Astra`
- 远端：`git@github.com:kbsonlong/Astra.git`，分支 `main`
- 本地基线：`b1c8e80`（2026-10-01，最后一次本地提交）
- 合并后 HEAD：`0421025`

---

## 一、拉取范围

本地 `main` 落后 `origin/main` **2 个提交**，远端共 **63 个文件变更（+4882 / −452）**：

| 提交 | 说明 |
|---|---|
| `267b76c` | `feat(tts): add multi-engine voice library and session routing` |
| `0421025` | `feat: 完成阶段一缺陷修复与架构优化` |

`0421025` 的主要内容：Nginx WebSocket 升级头、WS 清理重构、审校版本校验、音频探测超时、任务状态原子写、进程身份校验、前端 meeting 模块拆分，以及新增 `docs/improvement-plan.md`。

---

## 二、冲突识别

### 识别方法

先用 `git stash create`（不改动工作区）生成一个临时提交，再用 `git merge-tree --write-tree origin/main <stash-commit>` 做干跑三方合并。这样能在真正动工作区之前拿到精确的冲突文件列表。

### 结果：5 个文本冲突 + 17 个自动合并

| # | 文件 | 冲突性质 |
|---|---|---|
| 1 | `backend/app/api/ws_session.py` | 同一 `finally` 清理块，双方各自重写 |
| 2 | `backend/app/main.py` | `app_lifespan` 生命周期，双方各自插入不同清理逻辑 |
| 3 | `frontend/src/App.tsx` | 6 处：import、state 声明、`connect()`、`onopen`、按钮条件、状态面板 |
| 4 | `frontend/src/live/session.ts` | 双方在同位置各自新增导出 |
| 5 | `frontend/src/live/session.test.ts` | import 列表 + 各自新增测试块 |

实际执行 `git stash push -u` → `git merge --ff-only origin/main` → `git stash pop` 后，冲突文件与预测**完全一致**。

### 两侧差异概览

- **本地一侧**：长时通话可靠性改造（心跳与重连 PCM 缓冲、录音分段超限不中断、`end_session` 前提交 pending 音频、保留部分 PCM 供重连续写）、会议纪要 AI 配图（后端 `report_image.py` + `image_client.py` + `GET /{task_id}/report-image.png`，前端灯箱预览）、健康探测超时、`.env` LLM_IMAGE_* 配置、`vite.config` 动态 `BACKEND_PORT` 代理。
- **远端一侧**：TTS 多引擎路由（`tts_router.py` + `tts_voice_routes.py` + `frontend/src/tts/*` 语音库 UI）、WebSocket 握手中的 `voice_id` / `voice_revision` 锁定、任务 supervisor 租约、`reconcile_task` 缺陷修复、`process_identity` 列、前端 `frontend/src/meeting/*` 模块拆分。

---

## 三、逐文件处理方案

### 1. `backend/app/api/ws_session.py`

**关键决策 A：删除本地 `except WebSocketDisconnect: pass` 分支**，沿用远端移除 `WebSocketDisconnect` 导入（远端 `from fastapi import APIRouter, WebSocket`）。

- 依据：本文件在 `while True` 循环内用 `message["type"] == "websocket.disconnect"` 显式 break（`backend/app/api/ws_session.py:125`），根本不会抛 `WebSocketDisconnect`。Starlette 的 `WebSocket.receive()` 只在收到 `websocket.disconnect` 消息时改状态，不抛异常；`WebSocketDisconnect` 仅由 `receive_text` / `receive_bytes` 的 `_raise_on_disconnect` 抛出。本地该分支是死代码。

**关键决策 B：`finally` 块保留远端的 shield 结构，但把 `recording.finish()` 换成本地的 `recording.close()`。**

```python
finally:
    session.end()
    if generation_task is not None:
        generation_task.cancel()
        cleanup = asyncio.gather(generation_task, return_exceptions=True)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup)
        generation_task = None
    try:
        if recording is not None:
            # Keep partial PCM so a reconnect can resume the same recording;
            # only end_session finalizes the downloadable WAV.
            recording.close()
    finally:
        release = asyncio.create_task(limiter.release(client_ip))
        try:
            await asyncio.shield(release)
        except asyncio.CancelledError:
            await asyncio.shield(release)
```

- 依据：本地 `RealtimeRecording`（`backend/app/core/realtime_recording.py`）新增了 `close()`、缓冲写与 `FLUSH_BYTES`。`finish()` 会 finalize WAV 并删除 PCM 文件；本地需求是断线后**保留部分 PCM 供重连续写**（`test_websocket_reconnect_resumes_same_recording_and_heartbeat` 依赖：两段 PCM 共 320 帧）。`finish()` 内部也先调 `self.close()`，语义兼容。
- 远端的 `asyncio.shield` 保证 generation 取消与 `limiter.release` 必定完成，满足 `test_websocket_disconnect_cancels_generation_and_releases_ip_slot` 对 `app.state.audio_ip_limiter._active == {}` 的断言。

**合并后两侧特性均在位**

- 本地：`_take_pending_audio()`（`:94`）、`last_recorded_sequence` 去重（`:121/153/168`）、`audio_segment_limit_reached`（`:179`）、`heartbeat_ack`（`:274`）、`recording.close()`（`:226/361`）。
- 远端：`voice_id` / `voice_revision` 解析与 `tts_voice_store.snapshot()`（`:196-216`）、`TtsVoiceSnapshot` 转发（`:22-43`）。

### 2. `backend/app/main.py`

**合并后的 `app_lifespan`（`backend/app/main.py:259-306`）同时保留两侧生命周期：**

- 远端 supervisor 租约：`supervisor_id = uuid4().hex` → `try_acquire_supervisor` → `records = reconcile(...) if owns_supervisor_lease else processing_tasks()` → `if owns_supervisor_lease:` 块内 `clean_meeting_artifacts(...)` + `clean_expired_realtime_recordings(...)` → `supervisor = asyncio.create_task(_task_supervisor(app, supervisor_id))` → `try: yield finally:` 中 `supervisor.cancel()` + `asyncio.gather` + `await asyncio.to_thread(release_supervisor, supervisor_id)`。
- 本地双 worker 清理：`for worker_name in ("asr_worker", "meeting_asr_worker"): worker = getattr(app.state, worker_name, None); if worker is not None: await worker.aclose()`，替代远端只关 `asr_worker`。
- 本地 `app.state.meeting_asr_worker = AsrWorkerClient(meeting_asr, ...)` 保留，并接入 `MeetingPipeline(asr=app.state.meeting_asr_worker)`。

### 3. `frontend/src/App.tsx`（6 处）

| 位置 | 处理 |
|---|---|
| import（`:14-26`） | 保留双方：远端 `startSessionPayload` + 本地 `ReconnectPcmBuffer`，与 `acceptsGenerationEvent` / `formatElapsedTime` / `generationForNewConnection` / `HEARTBEAT_INTERVAL_MS` / `heartbeatExpired` / `nextActiveGeneration` 并列 |
| state 声明（`:204-213`） | 双方全部保留：`elapsedSeconds` / `reconnectCount` / `droppedAudioSeconds` / `micAccessIssue` / `microphoneDevices` / `selectedMicDeviceId` / `voiceChoices` / `ttsHealth` / `selectedVoiceId` / `sessionVoice` |
| `connect()` | 以远端为主干（麦克风权限申请 + `selectedSessionVoice()` 音色 pin 校验），但开头守卫换成本地更严的版本，避免重连中重复建连 |
| `openSocket().onopen` | 保留本地 `lastServerMessageAt.current = Date.now(); startHeartbeat();` 与 `finalizeAfterReconnect` 分支（`audio_format` → `flushReconnectFrames` → `end_session`），但把本地手写的 start_session 消息替换为远端的 `startSessionPayload(recordingId.current, sessionVoice.current)`（`:740-742`），使音色锁定参数随握手下发 |
| 按钮条件 | 统一用本地语义 `{!sessionActive ? (`（远端为 `{!connected ? (`）——本地有「断线重连中」中间态，用 `connected` 会误显示「开始通话」 |
| 状态面板（`:1113-1141`） | 四宫格合并为 6 格：会话状态=`connectionPhase`、当前阶段=`centerLabel`、录音时长=`formatElapsedTime(elapsedSeconds)`、输入设备=麦克风 label（远端）、自动重连=`reconnectCount` 次、未补传缺口=`droppedAudioSeconds`。CSS `.dial-meta` 为 2 列 grid，6 格自动排成 3×2 |

上游的 `.live-voice-summary` 区块（`:1162`）保留在「实时流」卡片头部，对应样式在 `frontend/src/styles.css:370`。

### 4. `frontend/src/live/session.ts`

双方在同位置各自新增导出，**保留双方**：远端 `VoiceSnapshot` 类型 + `startSessionPayload(recordingId, voice)`；本地 `HEARTBEAT_INTERVAL_MS` / `HEARTBEAT_TIMEOUT_MS` / `RECONNECT_AUDIO_BUFFER_SECONDS` + `ReconnectPcmBuffer` / `heartbeatExpired` / `formatElapsedTime`。

### 5. `frontend/src/live/session.test.ts`

import 列表同时含 `ReconnectPcmBuffer` 和 `startSessionPayload`；两个 describe 块均在（远端 "pins the selected voice revision in the session handshake" + 本地 "long-running session reliability"）。

### 自动合并的 17 个文件（无需人工干预）

- `frontend/src/UploadPage.tsx`：远端大重构（399 行变动，meeting 逻辑拆到 `frontend/src/meeting/*.ts`）与本地纪要图片灯箱 UI 自动共存。本地 `reportImageOpen` / `reportImageUrl` / `openReportImagePreview` / `image-lightbox` 全部保留（`:182-1089`），远端改用 `import { MeetingResult, OverlapDetection, waitForMeeting } from "./meeting/task"` 也已生效。
- `frontend/src/styles.css`：远端 `.mic-dialog*` / `.voice-*` / `.skeleton` + 本地 `.image-lightbox*`（`:581`）分区共存。
- `backend/app/config.py`：本地 `llm_image_model` / `llm_image_path` / `llm_image_size` / `llm_image_timeout_seconds` 与远端 TTS 路由、`audio_probe_timeout_seconds` / `audio_probe_temp_max_bytes` 字段级合并，`.env.example` 的 `LLM_IMAGE_*` 与 `AUDIO_PROBE_*` 同样共存。
- `backend/app/health.py`：本地 `HEALTH_PROBE_TIMEOUT_SECONDS = 2.0` + `asyncio.wait_for` 超时探测，远端 `_engine_status` 暴露路由字段。
- `backend/app/api/meeting_routes.py`：本地 `GET /{task_id}/report-image.png`（`:271`）与远端 `_write_json_atomic` / `_reap_meeting_process` / `meeting_status` / 版本校验共存。
- `backend/app/core/task_store.py`：**远端已修复本地进展报告的行动③**——`reconcile_task`（`backend/app/core/task_store.py:426`）改为先查 pid / 进程身份再判终态，新增 `_terminal_status()` 读 `status.json` 回填真实 done/failed；另增 `process_identity` 列、`try_acquire_supervisor` / `release_supervisor`、`terminate_and_wait`、`processing_tasks()`。
- 其余：`backend/app/core/realtime_recording.py`、`backend/app/core/session_manager.py`（新增 `has_pending_audio()`，超限不再清空缓冲）、`backend/app/schemas/ws.py`、`backend/tests/test_health.py`、`backend/tests/test_meeting_routes.py`、`backend/tests/test_realtime_recording.py`、`backend/tests/test_ws_session.py`、`.env.example`、`README.md`、`deploy/README.md`、`docs/acceptance-checklist.md`、`docs/architecture-analysis.md`、`docs/user-guide.md`、`frontend/vite.config.js`、`frontend/vite.config.ts`。

### 未跟踪新文件（已保留）

`Makefile`、`backend/app/core/report_image.py`、`backend/app/models/image_client.py`、`docs/progress-review-2026-10-28.md`。

---

## 四、验证结果

### 后端

| 项目 | 结果 |
|---|---|
| `pytest -q backend/tests`（合并后） | **9 failed, 220 passed** |
| `pytest -q backend/tests`（上游基线 0421025，隔离 worktree） | **9 failed, 213 passed** |
| `pytest backend/tests/test_ws_session.py backend/tests/test_realtime_recording.py` | **17 passed** |
| `ruff check backend` | All checks passed! |
| `mypy`（项目配置，4 文件 scope） | Success: no issues found in 4 source files（上游同） |

**关于 9 个失败用例**：合并后与上游基线的失败用例**完全相同**，根因统一为沙箱文件权限：

```
PermissionError: [Errno 1] Operation not permitted:
'/Users/zengshenglong/.astra/realtime-recordings/1ab4b731-777e-4077-aade-40f751bf6ffc.pcm'
```

测试进程需要写入 workspace 之外的 `~/.astra/`，被沙箱拒绝。**结论：9 个失败是环境权限问题，不是合并引入的回归**；合并后比上游基线多 7 个通过用例。

补充：全量 `mypy backend/app` 报 84 errors，是仓库既有状态（上游同命令 86 errors，合并后反而少 2 个），与本次合并无关。

### 前端

| 项目 | 结果 |
|---|---|
| `npx tsc -b` | 无错误 |
| `npm test`（vitest） | **8 files / 21 tests passed** |
| `npm run build` | 成功（`dist/index.html` 1.21 kB、`index-D6uRKL9T.css` 43.56 kB、`index-CRro7bcj.js` 344.21 kB） |

### 全局检查

- 全仓扫描 `.py/.ts/.tsx/.js/.md/.css/.yml`：**无任何冲突标记残留**。
- `git diff --check`：无输出（无空白/冲突残留问题）。
- `git diff --name-only --diff-filter=U`：空（5 个冲突文件全部标记为已解决）。

---

## 五、遗留事项与建议

### 需要人工确认的口径差异

1. **音频分段超限的协议语义不一致（重要）**。上游原逻辑是超限时发送 `audio_too_large` 并 `websocket.close(1009)` 关闭连接；本地改为发送 `audio_segment_limit_reached` 并**保持连接**。合并后 `backend/tests/test_ws_session.py` 中上游的 `test_websocket_rejects_audio_over_session_limit` 被本地的 `test_websocket_keeps_recording_session_alive_when_segment_limit_is_reached` 覆盖，当前工作区沿用本地语义。前端 `frontend/src/App.tsx` 已同时兼容两种错误码（`event.code === "audio_segment_limit_reached" || event.code === "audio_too_large"`）。**建议：明确以哪一侧为准，并同步后端协议与测试。**
2. **PRD FR-12 口径未同步**。`docs/product-requirements-document.md:150` 要求 FR-12 图片生成「必须本机完成」，但本地实现复用远端 `llm_base_url` + `LLM_IMAGE_MODEL=codex/gpt-image-2.5`；同文件 `:169` 仍写 FR-12「规划中」，而代码已实现（`backend/app/api/meeting_routes.py:239`）。

### 工程卫生

3. **`frontend/vite.config.js` 是误跟踪的编译产物**。仓库同时跟踪 `vite.config.ts`（源码）、`vite.config.js`（tsc 编译输出）、`vite.config.d.ts`。本地已把 `.js` 同步为新编译结果，但**建议删除 `.js` 与 `.d.ts` 并加入 `.gitignore`**（该问题在 `b1c8e80` 与 `0421025` 中均存在）。合并期间 `frontend/vite.config.d.ts` 与 `frontend/tsconfig.node.tsbuildinfo` 已被 `git checkout --` 还原为编译生成状态。
4. **`git stash@{0}: On main: astra-local-wip-2026-10-28` 仍保留**，作为保险。确认合并结果无误后可 `git stash drop`。
5. **合并尚未提交**。当前 26 个文件已 staged、4 个未跟踪文件待 add。建议拆成两个提交：①合并上游 2 个提交 + 冲突解决；②本地增量功能（会议纪要配图、长时通话可靠性）。
6. **`resemblyzer` 声纹聚类仍然失效**：`pkg_resources` 已被 setuptools 82 移除，`resemblyzer==0.1.4` 需要回退或改写（此项来自 2026-10-28 进展报告，本次合并未涉及）。
7. **`.env.example` 与 `.env` 漂移**需收敛（本次合并已让两套新增配置共存于同一 `config.py`，但实际运行环境的取值仍需确认）。

---

## 六、操作留痕

| 步骤 | 命令 | 结果 |
|---|---|---|
| 备份 | `git diff` → `/tmp/astra-backup/local-working.patch`（90548 字节）+ `untracked-1793197183.tar.gz` | 完成 |
| 暂存本地改动 | `git stash push -u -m "astra-local-wip-2026-10-28"` | 完成 |
| 快进到远端 | `git merge --ff-only origin/main` | HEAD = `0421025` |
| 恢复本地改动 | `git stash pop` | 5 个文件冲突（与预测一致） |
| 解决冲突 | 逐文件语义合并 | 5/5 完成 |
| 标记已解决 | `git add <5 个文件>` | `diff-filter=U` 为空 |
| 基线对照 | 隔离 worktree `/tmp/astra-upstream` @ 0421025 | 已清理（`git worktree remove`） |
