# 会议纪要增强：引用锚点 + 断点续跑 + 纪要任务并行 实施方案

> 参考：飞书(Lark)桌面端会议纪要实现拆解（REA 逆向结论）的三条核心设计。
> 日期：2026-11-03  ·  状态：已确认，按 F1 → F2 → F3 顺序实施

## 背景与目标

对照飞书会议纪要的三个核心设计，补齐 Astra 会议管线的差距：

| # | 飞书设计 | Astra 现状 | 差距 |
|---|---------|-----------|------|
| F1 | 纪要图表分支绑原句，点回原话，统计引用数 | Segment 有时间戳/speaker，但纪要是纯文本、无 segment 引用 | 纪要输出挂 segment 锚点 |
| F2 | 录音切片上传成功才删，崩溃后自动补传 | meeting job 崩溃后 reconcile 只收敛状态，必须整条重跑 | ASR 产物 checkpoint + 断点续跑 |
| F3 | 总结/待办/决策等任务并行、各自超时、先完先显示 | SummaryStage → TranslationStage 严格串行 | 纪要子任务 fan-out |

## 总体设计

三个特性都落在现有 job 化骨架内（`meeting_cli.py` 子进程 + `status.json` + WS 轮询推送），不改 WS 协议：

```text
POST /api/meeting/process [--resume]
  └─ meeting_cli 子进程
       ├─ [F2] resume? 读 segments.json checkpoint，跳过解码+ASR
       ├─ transcribe → [F2] 写 segments.json checkpoint（ASR 完成即落盘）
       ├─ [F3] 并行: summary | actions | decisions（各自 timeout，完一个写一次 status.json）
       ├─ [F3] translation（依赖 summary，summary 完成后跑）
       ├─ [F1] citations = 正则提取 summary 中的 [seg-XXXXXX] 引用 + 校验
       └─ report.md / transcript.txt / meta.json / status.json
WS /api/meeting/{task_id}/events —— 每秒轮询 status.json 全量推送（现状已支持，零改动）
```

---

## F1 纪要引用锚点

**目标**：AI 纪要每个要点标注来源段落 ID；报告渲染引用统计；引用可反查原句。

**改动**：

1. `backend/app/core/workflow.py`：`Segment` 增加 `segment_id: str = ""` 字段（默认空，兼容旧调用）。
2. `backend/app/core/meeting.py`：
   - 新增 `_assign_segment_ids(segments)`：转写完成后按顺序赋 `seg-{i:06d}`（普通与 separation 路径都覆盖，在 `process()` 里统一赋，保证 timeline / prompt / 引用三方一致）。
   - `_generate_summary()`：逐字稿行改为 `[seg-000001][S1] 文本`；system prompt 末尾追加固定指令行："每个要点末尾标注来源段落 ID，格式 [seg-XXXXXX]，可多个。"（追加而非改模板，自定义模板不受影响）。
   - 新增 `extract_citations(summary) -> dict[str, int]`：正则 `\[(seg-\d{6})\]` 统计。
   - `MeetingResult` 增加 `citations: dict[str, int]`；`process()` summarize 后填充。
3. `backend/app/api/meeting_routes.py` `_render_markdown()`：summary 之后追加 `## 🔗 引用统计`：每个被引 seg id + 次数 + 原句摘录（≤80 字）；列出 summary 中引用了不存在 seg id 的"无效引用"条目（说空话检测的雏形）。
4. `transcript.txt` 格式升级：`[mm:ss] [seg-000001] S1 文本`（`timeline_text()`）。

**测试**：
- `test_workflow.py`：timeline 含 segment id 断言更新。
- 新增：`extract_citations` 正常/空/无效 id；`_render_markdown` 引用段渲染；无效引用标记。
- 回归：现有 `test_meeting_routes.py` 渲染断言更新。

**验收**：report.md 的 AI 摘要每个要点可追溯到 `[seg-XXXXXX]`，引用统计段无无效引用时显示"无"。

---

## F2 断点续跑

**目标**：ASR（最耗时阶段）完成后立即 checkpoint；job 失败后重试只跑 LLM 阶段。

**改动**：

1. `backend/app/core/meeting.py` `process()` 增加参数：
   - `checkpoint_dir: str | Path | None = None`：转写完成后原子写 `segments.json`（segment_id/start/end/speaker/speaker_name/text，不含音频字节）。
   - `resume: bool = False`：为真且 checkpoint 存在时，直接加载 segments 跳过解码+ASR（diarization/correction 结果已固化在 checkpoint 里）。
   - checkpoint 带版本号 `{"version": 1, "filename": ..., "segments": [...]}`，resume 时校验 filename 一致、版本支持。
2. `backend/app/core/meeting_cli.py`：
   - `--resume` flag，透传 `resume=True, checkpoint_dir=out_dir`。
   - status.json 增加 `"resumed": bool`。
   - 已有 `status: failed` + 有效 checkpoint 时 `--resume` 才有意义；checkpoint 缺失/损坏时报错退出并写 failed。
3. `backend/app/api/meeting_routes.py`：
   - `POST /api/meeting/process` 增加 `resume: bool = Form(False)`；`resume=True` 时校验 job 目录存在 checkpoint，否则 400。
   - 透传给子进程 argv。
4. 清理语义保持不变：成功后 checkpoint 保留（供审计/重放），`artifact_retention` 统一管理。

**测试**：
- 新增 `test_meeting_resume.py`（或并入现有）：checkpoint 写入/加载 round-trip；resume 跳过 ASR（fake pipeline 断言 transcribe 未调用）；filename 不匹配拒绝；checkpoint 损坏 → failed；routes `resume=True` 无 checkpoint → 400。
- 回归：`test_task_store` reconcile 语义不变。

**验收**：杀掉运行中的 meeting job → 带 `--resume` 重跑 → ASR 零调用，直接出纪要；status.json `resumed: true`。

---

## F3 纪要子任务并行（fan-out）

**目标**：总结 / 待办 / 关键决策三个任务并发执行、各自独立超时；谁先完成谁先写 status.json，WS 事件先推谁；翻译依赖总结，总结完成即启动。

**改动**：

1. `backend/app/core/meeting.py`：
   - 新增 `NOTE_TASKS = ("summary", "actions", "decisions")` 与各任务内置 prompt（待办：`[ ]` 行动项格式；决策：决策 + 依据 + 反对意见）。
   - `summarize()` 重构：三个任务 `asyncio.gather` 并发，每个任务包 `asyncio.wait_for(timeout)`；单任务超时/失败不拖垮其他任务（结果标 `failed/timeout`，其余照常）。
   - 翻译在 summary 落定后启动（成功或失败文本都翻译 summary 可用部分）。
   - `MeetingResult` / `MeetingContext` 增加 `note_tasks: dict[str, dict]`：`{status: done|failed|timeout, text, elapsed_s, error?}`；`summary`/`translation` 字段保留（兼容下游）。
   - 进度回调：每个任务完成时调用 `on_note_task(task, payload)`（meeting_cli 用它更新 status.json）。
2. `backend/app/core/meeting_cli.py`：`on_note_task` 回调里原子更新 status.json 的 `note_tasks` 字段 → WS events 每秒轮询自动推送（协议零改动）。
3. `backend/app/config.py`：`meeting_note_task_timeout_seconds`（默认 300）。
4. `backend/app/api/meeting_routes.py` `_render_markdown()`：按 note_tasks 渲染 `## ✅ 待办` `## 🧭 关键决策`；失败/超时任务显示占位说明。

**测试**：
- 新增：三任务并发（fake LLM 不同延迟，断言先完成者先回调）；单任务超时隔离；translation 依赖时序；note_tasks 渲染；CLI 回调写 status.json。
- 回归：现有 summary-only（`--no-summarize`）与无 llm 路径不变。

**验收**：长会议录音处理时，status.json/WS 中 note_tasks 逐个从 processing → done；任一 LLM 任务超时不影响其他任务出结果。

---

## 实施顺序与提交切分

| 提交 | 内容 |
|------|------|
| 1 | 本方案文档 |
| 2 | F1：segment id + prompt 引用 + citations + 渲染 + 测试 |
| 3 | F2：checkpoint + resume（CLI/routes）+ 测试 |
| 4 | F3：fan-out + per-task timeout + status.json 进度 + 测试 |
| 5 | docs 收尾（README 会议章节补充 resume/引用说明） |

每个提交独立可回归：`PYTHONPATH=backend .venv/bin/pytest -q backend/tests`。

## 已知风险

- ~~`mlx-tune==0.6.0` 与 `numpy==1.26.4` metadata 冲突~~ 已解决（2026-11-03，以新版为准）：numpy 升 `2.2.6`、funasr 升 `1.4.21`（放开 numpy<2 上限）、constraints 加 `mlx-vlm<0.6.0`（避开 starlette>=1.0.1 与 fastapi 0.116.1 的冲突）；`uv pip install -r requirements-dev.txt` dry-run 解析通过，242 测试 + resemblyzer 真实模型加载在 numpy 2.2.6 下全过。
- timeline 格式变更会破坏旧 report 的文本比对（无机器消费方，接受）。
- transcript_segments.jsonl 的 seg id 基于**合并后**训练段，与 timeline 的 raw segment id 不同域——保持现状不合并，报告引用一律指向 timeline id，训练标注继续用 JSONL id，两域在 meta.json 中注明。
