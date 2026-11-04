# Astra 项目进展分析（数据截至 2026-10-28）

> 本报告基于仓库登记数据 + 运行时实测得出。不编造数据中不存在的事实；信息不足处明确标注「数据不足」。

---

## 1. 当前状态判断

**代码层面在正轨，运行与登记层面已经偏了。**

**在正轨**：工程基线扎实——Ruff/Mypy 全绿、后端 200 个单测通过（5 个失败为沙箱权限，非代码缺陷）、前端 build/test 通过；架构、PRD、设计系统文档齐全；`docs/code-review-2026-09-10.md` 的 14 项里 P0-01~04、P1-01~05、P2-01、P2-02 已由 git log 证实落地，前端令牌层 + 声纹罗盘 + 时间脊柱也已进代码。

**已经偏了**：**登记机制整体失效**——三份清单共 92 个 checkbox（`tasks.md` 9 个、`checklist.md` 54 个、`acceptance-checklist.md` 29 个）**0 个勾选**；**项目实际停摆**——最后一次 commit 2026-10-01，至今 **27 天零提交**，工作区未提交改动停在 2026-10-14（14 天未动），任务库 `~/.astra/tasks.sqlite3` 中 10-15 之后**零任务运行**；**运行时已有可观测的静默降级**——最近两次会议任务都因 `pkg_resources` 缺失跳过声纹分离，`speakers: []`。

---

## 2. 时间风险

> ⚠️ **「剩余时间」这一维度：数据不足。** 全部登记数据中**没有任何**任务截止日期、里程碑日期、负责人或发布窗口（`grep` 覆盖 docs/、.trae/、design/、.workbuddy-ai/ 全部无命中）。因此下面用**唯一可用的时间轴——各计划文档的发布日至今**做代理推算，并把「已过天数」与「完成度」的匹配情况列出。真正的时间风险结论需要补齐日期后才能定（见第 5 节）。

| 计划项 | 基准日 | 已过天数 | 完成度（可验证） | 匹配判断 |
|---|---|---|---|---|
| .trae 局域网语音助手 MVP（9 Task / 54 Checkpoint） | 2026-08-28 | 61 天 | 0/54 勾选 | **最可能来不及，但更可能是"已作废未归档"**——见下方判断依据 |
| 改进路线图 P0/P1/P2（14 项） | 2026-09-10 | 18 天 | 11/14 已实现，P2-03/04/05 未动 | **偏慢但可追**：剩 3 项均为中等工作量 |
| 界面设计落地三步 | 2026-09-17 | 41 天 | ①令牌层 ✅ ②UploadPage 拆分 ❌ ③声纹罗盘/时间脊柱 ✅ | **偏慢**：②未做且文件反增（1257→1370 行） |
| IndexTTS 2.5 MLX 接入（L0→L6） | 2026-09-19 | 39 天 | L0 ✅ L1 ✅ **L2 ❌** L6 脚本有、未跑通 | **最可能来不及** |
| VoiceStudio 路线图 P0/P1/P2 + 真机验收 | 2026-09-17 | 41 天 | 代码 ✅，**真机验收 0 记录** | **验收证据缺口最大** |
| PRD 第 12 节 P1 近期建议 5 条 | 2026-09-10 | 18 天 | 登记数据中**无任何完成证据** | **大概率偏慢**（缺证据 ≠ 未做，但无登记） |
| FR-12 会议配图 | 2026-09-10 | 18 天 | 工作区**已实现**（未提交） | 正轨，唯一问题是与 PRD 冲突 |

**最可能来不及的三个（附判断依据）**：

1. **IndexTTS MLX 成为默认 TTS 后端** —— 依据 `docs/indextts-2.5-analysis-and-mlx-validation.md:462` 记载 L2 本轮未完成：隔离缓存下载 20 分钟仅得 1.5GiB、磁盘只剩 15GiB 即因速度与空间风险中止，**未生成输出 WAV、未跑 ffprobe**；`~/.astra/models/indextts-2.5-mlx` 目录至今不存在。同时 `docs/indextts-2.5-mlx-adapter-design.md:294` 明确「**在 L2 未通过前，不能把 indextts_mlx 设为 Astra 默认后端**」。当前磁盘 460Gi 用 410Gi（**97%，剩 16Gi**），而 `~/.cache/huggingface` 已占 8.6G——**这个阻塞目前无解**，除非先腾空间。
2. **真机验收清单 29 项** —— `docs/acceptance-checklist.md` 29 项全未勾选，且**不存在任何验收记录文件**。这项需要"人拿设备跑"，是人工密集型工作，而项目已 27 天无提交，属于典型的"没人推进就不会动"。
3. **.trae MVP 54 项** —— 若按字面执行，61 天 0% 完成必然来不及；但必须指出：**该方案要求 4 个容器 + whisper.cpp + Piper HTTP + 192.168.3.18 LLM**（`.trae/specs/astra-voice-assistant-mvp/checklist.md` Checkpoint 1.1/1.2），而 `deploy/` 下只有 README.md、无 `deploy/docker` 与 `deploy/models`，docker-compose 仅 frontend-nginx 一个服务；`AGENT.md` 明确「MLX、Piper 和真实模型只在 Apple Silicon Mac mini 原生运行，**不要把后端强行塞进 Linux Compose 容器**」。**结论：这份清单的 0% 主要是"计划已被架构演进取代但没归档"，不能当作真实欠账。**

---

## 3. 阻塞与依赖

### 被外部条件卡住

| 阻塞 | 证据 | 影响 |
|---|---|---|
| **磁盘 97% 满（剩 16Gi）** | `df` 实测 460Gi/410Gi | 卡死 IndexTTS L2 模型下载（硬阻塞） |
| **`.env.example` 默认 LLM 不可达** | `curl http://192.168.3.18:8000/v1/models` → `curl: (7) Failed to connect ... after 4062 ms` | 照默认配置部署必然起不来 |
| **`pkg_resources` 已不存在** | `.venv` 有 setuptools 82.0.1，但 `import pkg_resources` → `ModuleNotFoundError`；setuptools 82 已移除该模块 | `resemblyzer==0.1.4`（`backend/requirements.txt:15`）内部 import 它 → 声纹聚类 100% 失效 |
| **Piper 语音模型缺失** | `backend/app/config.py:295` 默认 `tts_model_path="models/zh_CN-huayan-medium.onnx"`，而 `models/` 下只有 `silero_vad.onnx` + 2 个 sherpa-onnx 目录 | TTS 兜底链在本机不可用 |

### 被代码缺陷卡住（自己造成，非外部）

- **`backend/app/core/task_store.py:285-300` `reconcile_task` 判定顺序错误**：先判 timeout 再判 `pid_alive`，timeout 分支直接 terminate。**证据**：任务 `20261014-114118-8208c09d` 的 `~/Astra/meetings/20261014-114118-8208c09d/status.json` 实际是 `{'status':'done','duration_s':2322.4,'segments':131}`，job.log 末尾也打印 `{"status": "done", ...}`，任务库却记为 `failed / task timed out after 7200s`。**即：有一半"失败"是误报，真实成功率比账上高。**
- **`backend/app/core/workflow.py:601`** 对声纹失败只 `logger.warning("diarization failed, skip: %s")` 后静默返回——降级无声，任务照样标 done，**这正是 `speakers: []` 却显示成功的原因**。

### 被"人"卡住

- **停滞 27 天**：最后一次 commit 2026-10-01（`b1c8e80`），`git log --since=2026-10-01` 返回 0 条；工作区 25 个已跟踪文件改动（+1133/-145）+ 3 个新文件（`Makefile`、`backend/app/core/report_image.py`、`backend/app/models/image_client.py`）**未提交已 14 天**。
- **文档与实现失同步**（都是"没人维护登记"造成的）：
  - `docs/product-requirements-document.md:169` 仍写 FR-12「规划中，当前代码未实现」，但代码已实现（`backend/app/api/meeting_routes.py:239`），且 20261014 任务目录确实产出了 `report-image.png`。
  - 更严重：同文档 `:150` 要求「**该流程必须在本机完成，不把提示词、会议文本或生成图片发送到外部图片服务**」，而实现复用 `llm_base_url` + 默认 `LLM_IMAGE_MODEL=codex/gpt-image-2.5`，指向远端 192.168.3.18 —— **这是 PRD 与实现的实质性合规冲突，需要裁决口径**。
  - `design/README.md:112` 写「UploadPage.tsx（1010 行）」，实际已 1370 行；`:118-120` 要求「生产环境必须自托管这三个字体文件（Latin 子集合计约 120KB），产品跑在局域网里，不能假设有外网」，但 `frontend/index.html:7` 仍外链 `fonts.googleapis.com`、`frontend/public` 不存在、无任何 .woff 文件。
  - `.env` 仅 16 键 / `.env.example` 89 键，本地缺 `ASTRA_CONFIG_PATH / TTS_MODEL_PATH / ADMIN_TOKEN / TTS_BACKEND / ASR_BACKEND / REALTIME_RECORDING_DIR / TASK_STORE_PATH`，又多出 7 个 example 里没有的键。
  - `frontend/vite.config.js`（1062B）是 `vite.config.ts` 的编译产物重复件，误入源码树。

### 不构成阻塞但会被误判

- pytest 5 个失败是沙箱对 `~/.astra/realtime-recordings/*.pcm` 的 `PermissionError: [Errno 1] Operation not permitted`（已用探针复核：该目录连新建文件都被拒），**属环境权限，不是产品缺陷**——但会让人误以为 CI 红了。

---

## 4. 下一步动作

> 由于登记数据中没有期望的交付日期，下面「建议完成时间」是按依赖紧迫度提出的**建议值**，请按实际窗口调整。

**① 解除磁盘阻塞，并让 IndexTTS L2 真正跑通** — 目标：IndexTTS MLX 默认化路径
先确认哪些目录可删（`~/.cache/huggingface` 8.6G 中的重复 ASR 权重、`~/Astra/meetings` 587M、仓库 `recordings/` 398M 中 3 段超长导出 WAV 3622s/3410s/5362s），把可用空间拉到 **≥ 20GiB**；再按 `docs/indextts-2.5-analysis-and-mlx-validation.md` 重跑 L2，产出输出 WAV 并通过 ffprobe，随后执行 L6。
**建议完成时间：2026-11-04（7 天）**。这是所有 TTS 相关项的前置，不做这一条其余 TTS 验收都无意义。

**② 修 `pkg_resources` 依赖，让声纹分离不再静默失效** — 目标：会议声纹分离闭环
在 `backend/requirements.txt` 显式钉住 `setuptools<81`（或改用不依赖 `pkg_resources` 的声纹实现），重装后跑一次真实音频会议，确认 `status.json` 的 `speakers` 不再是 `[]`。
**建议完成时间：2026-10-30（2 天）**。这是单点依赖问题，但已连续 2 个任务静默降级（20261014-110304、20261014-114118），拖得越久污染越多。

**③ 修 `reconcile_task` 判定顺序，并回填误报状态** — 目标：任务状态可信
把 `pid_alive` 检查提到 timeout 之前（或 timeout 时先读 `status.json` 再决定），然后把 `20261014-114118-8208c09d` 按实际 `done` 回填。顺带给 workflow 的声纹降级加一条显式告警字段，别让"降级"长得像"成功"。
**建议完成时间：2026-10-30（2 天）**。它直接决定后面所有"成功率"判断是否可信。

**④ 收敛环境配置漂移，让"照文档启动"真的能起来** — 目标：可复现部署
对齐 `.env` 与 `.env.example`（补齐 `ASTRA_CONFIG_PATH / TTS_MODEL_PATH / ADMIN_TOKEN / TTS_BACKEND / ASR_BACKEND / REALTIME_RECORDING_DIR / TASK_STORE_PATH`，删掉 7 个野键）；补装 Piper 语音模型到 `models/zh_CN-huayan-medium.onnx`；决定 192.168.3.18 是废弃还是恢复（本地 `127.0.0.1:3425` 有 `magpie` 监听且 `/v1/models` 可返回，可用），并把 `LLM_BASE_URL` 写进文档。
**建议完成时间：2026-10-31（3 天）**。

**⑤ 提交工作区存量改动，清理重复文件，并同步文档口径** — 目标：仓库可追溯 + 登记恢复
提交 25 个改动 + 3 个新文件（含 `report-image.png` 端点、`HEALTH_PROBE_TIMEOUT_SECONDS=2.0`、Makefile `make dev`）；删除 `frontend/vite.config.js` 重复件；把 PRD FR-12 状态从"规划中"改为"已实现"，并**同时写清"本地生成 vs 远端图片服务"的最终口径**；把 `.trae` 旧 54 项 MVP 清单标注为"已被原生 Mac 架构取代/归档"，别让它继续污染进展统计；字体自托管（3 个字体 Latin 子集约 120KB）并拆 `UploadPage.tsx`（P2-04）。
**建议完成时间：2026-11-03（5 天）**，其中"提交 + 删重复件 + PRD 状态更新"建议 **10-31 前**先做掉（零成本、直接消除 14 天的搁置风险）。

---

## 5. 需要补充的信息

以下是判断时**需要但登记数据里没有**的内容，补齐后可再出一版带真实时间轴的风险排序：

### 时间与责任（缺失最严重，直接导致第 2 节无法真正成立）

1. **每个项目的目标交付日期 / 里程碑节点**（如"IndexTTS 默认化必须哪天完成""真机验收窗口是哪周"）——目前 92 个 checkbox 全部无期限，无法判断"来不及"。
2. **各项工作的负责人**——无法区分"没做"是"没排期"还是"人不在"。
3. **验收标准**：29 项真机验收里哪几项是"必须通过才算完成"的硬门槛？哪些是可选记录项？

### 口径与范围

4. **`.trae` 旧 MVP 方案（4 容器 + whisper.cpp + Piper HTTP + 192.168.3.18）是否正式作废**？还是仍要保留为一条部署路径？这决定 54 项 checklist 是"欠账"还是"垃圾"，直接影响风险排序。
5. **PRD FR-12「必须本机完成」是硬约束还是早期设想**？当前实现走远端图片服务且已产出真实图片——是要改成本地模型，还是要改 PRD？
6. **IndexTTS 是否仍是默认 TTS 后端的必选项**，还是允许长期保持 `indextts_mlx` 为可选、piper/其他兜底？若可降级，动作 ① 的优先级可以下调。
7. **最终部署形态**：原生 Mac mini 单机，还是确实要 Docker 多服务？这决定 `.env` / `deploy/` / nginx 的收敛方向。
8. **Qwen3-ASR 微调**（`docs/qwen3-asr-finetuning-design.md` 结论是 Mac mini/MPS 非官方路径、生产训练需 CUDA）——是否在本次范围内？这直接决定要不要单独采购 CUDA 机器。

### 资源与运维

9. **磁盘清理授权**：哪些目录/文件可以安全删除？`~/.cache/huggingface` 8.6G 里 mlx-community/Qwen3-ASR-0.6B-4bit 与 1.7B-8bit 是否两套都要留？
10. **是否可扩容**（外接盘 / 换盘 / 云端模型缓存）？若不扩容，IndexTTS 全量下载这条路在物理上就走不通。
11. **目标使用规模**（局域网人数、并发会议数、典型音频时长）——PRD 第 10 节写明"当前 MVP 不设定生产级并发和延迟 SLA"，但没有这些数就无法定性能基线。
12. **会议录音与纪要的留存期/合规要求**——`recordings/` 398M + `~/Astra/meetings` 587M 持续增长，另有 `artifact_cleanup_events` 仅 2 条记录。

---

**一句话总结**：代码质量在正轨，但项目已经停了 27 天，且被三个可立即解决的具体问题卡住——磁盘 97% 满（卡死 IndexTTS）、`pkg_resources` 缺失（声纹静默失效）、`reconcile_task` 判定顺序（把成功误报成失败）；更关键的是 92 个 checkbox 零勾选，**进展根本不可见**，需先补上日期与负责人，第 2 节的时间风险才能给出真实结论。
