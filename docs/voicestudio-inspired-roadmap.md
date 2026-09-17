# VoiceStudio 借鉴路线图

本计划基于对 [VoiceStudio](https://github.com/debpalash/VoiceStudio) 源码的分析，结合 Astra
当前实现（`backend/app/models/`、`backend/app/core/workflow.py`、`backend/app/main.py`）制定。
目标是补齐 Astra 与成熟中文语音项目之间的关键差距，同时不偏离 Astra「局域网中文会议 /
语音助手」的聚焦定位。

## 背景：现状与差距

Astra 现状（源码事实）：

- ASR：`MlxAudioAsrClient`，底层 Qwen3-ASR（MLX），`asr_language="Chinese"` 默认，
  已支持 hotwords、system_prompt、长音频分块、prompt 泄漏切除、线程本地模型缓存。
- TTS：`PiperSdkTtsClient`，单一 Piper 声（`models/zh_CN-huayan-medium.onnx`），
  无克隆、无情感、无多音色，是当前最薄弱环节。
- Workflow：`VAD → ASR → 标点恢复 → 说话人分离`，协议注入式；已有声纹注册、
  会议纪要、审校、训练数据导出、ZipEnhancer/JAEC/重叠检测等前处理。
- 引擎为硬编码单实现，ASR/TTS 无能力声明与可用性探测，无法并存切换。

VoiceStudio 验证过、值得借鉴的实践：

- 默认 OmniVoice 模型中文语料 111,343 小时（全语种第 2），粤语 13,302 小时。
- CosyVoice 3 / GPT-SoVITS 在源码中显式带 `<|zh|>` 与 `<|yue|>` 标记，支持零样本克隆。
- IndexTTS 2.5 将 `zh` 列为支持语言首位，含情感控制，一键 sidecar 安装。
- FunASR（SenseVoice）通过 cam++ 模型做 ASR 内联说话人分离，输出 `Speaker N` + 毫秒时间戳。
- 各引擎统一暴露 `supported_languages` 与 `is_available() -> (bool, reason)`，实现优雅降级。

## 计划总览

| 优先级 | 方向 | 目标 | 状态 |
|:--:|------|------|:--:|
| P0 | 方向一：中文原生 TTS | 用 CosyVoice/MLX-Audio 替换/并存 Piper，获得中文+粤语+克隆 | ✅ 已落地 (MlxAudioTtsClient) |
| P1 | 方向二：引擎可插拔抽象 | 为 ASR/TTS 增加能力声明与可用性探测，支持多后端并存 | ✅ 已落地 (capabilities/is_available + 工厂 + health) |
| P2 | 方向三：会议内联说话人分离 | 引入 FunASR/SenseVoice 后端，评估转写+分离一体化 | ✅ 已落地 (FunAsrClient + InlineAsrDiarizeStage) |

落地记录：

- P0：新增 `MlxAudioTtsClient`，`tts_backend` 可在 piper/mlx_audio 间切换，Piper 兜底保留。
- P1：`MlxAudioAsrClient`/`PiperSdkTtsClient`/`MlxAudioTtsClient`/`AsrWorkerClient` 均实现
  `capabilities()` 与 `is_available()->(bool,reason)`；`_build_realtime_asr`/`_build_tts_client`
  工厂化；`/api/health` 暴露引擎 mode/available/reason/capabilities，对旧 client 优雅降级。
- P2：新增 `FunAsrClient`（SenseVoice + cam++ 内联分离，`normalize_funasr_segments` 解析
  `sentence_info`/`spk`/富标签清洗）；`workflow.py` 新增 `InlineAsrDiarizeStage` 与
  `SpeakerRegistryMappingStage`，`AudioWorkflow` 的 `inline_asr` 分支用「整段转写+分离」取代
  VAD+ASR+SD 并把 `Speaker N` 映射到已注册声纹；`asr_backend=funasr` 经 `meeting_cli` 装配启用。

真实模型验收（Mac mini / funasr 环境）仍需在硬件上单独执行，见下方各方向验收标准与末尾说明。

优先级理由（历史，供参考）：方向一直接决定语音助手体感且改动被 `tts_client.py` 协议隔离，风险最低、收益最大，
故列 P0。方向二是架构性投资，为后续多引擎并存铺路，且方向一落地后再抽象更有依据。方向三收益
明确但需先有方向二的引擎抽象承载，故列 P2。

---

## 方向一（P0）：中文原生 TTS 接入

### 目标
在保留 `PiperSdkTtsClient` 作为兜底的前提下，新增至少一个中文原生 TTS 后端，使语音助手输出
具备中文/粤语自然度，并为后续声音克隆/情感能力打基础。

### 候选与取舍
- **CosyVoice 3**：中文质量天花板，带 `<|zh|>`/`<|yue|>`，支持零样本克隆；官方推理路径偏 CUDA，
  Mac mini 上需验证 CPU/MPS 可行性与延迟。
- **MLX-Audio（Kokoro 等）**：与 Astra 现有 MLX 技术栈一致，Apple Silicon 原生、部署最简，
  中文自然度中等；建议作为 Mac mini 首选落地项。
- **IndexTTS 2.5**：`zh` 首位 + 情感控制 + 一键 sidecar，作为第二阶段增强项。

建议：**先接 MLX-Audio 走通协议与 Mac mini 验收，再评估 CosyVoice 提升质量**。

### 步骤
1. 在 `backend/app/models/tts_client.py` 新增 `MlxAudioTtsClient`，实现现有契约
   （`is_ready()`、`async synthesize(text) -> bytes`），复用 MLX 线程本地约定（参考
   `asr_client.py` 的 `_run_mlx` / 模型缓存做法）。
2. 在 `backend/app/config.py` 增加类型化配置：`tts_backend`（`piper` | `mlx_audio`）、
   `tts_mlx_model` 等，并在 `Settings.from_env()` 用现有 `_*_env` 辅助读取；同步更新 `.env.example`。
3. 在 `backend/app/main.py` 的 `create_app()` 构建区，按 `tts_backend` 选择实例化的 TTS client。
4. 补单元测试：仿照 `backend/tests/test_tts_client.py`，用注入 loader 覆盖合成与错误路径。

### 验收标准
- `PYTHONPATH=backend .venv/bin/pytest -q backend/tests/test_tts_client.py` 通过。
- Mac mini 上以真实模型合成一段中文文本，产出非空可播放音频（人工验收，记录延迟）。
- `tts_backend=piper` 时行为与现状完全一致（回归不破坏）。

---

## 方向二（P1）：引擎可插拔抽象

### 目标
为 ASR/TTS client 引入轻量能力声明与可用性探测，使多后端可并存、可切换、可向前端解释「为何不可用」。

### 步骤
1. 定义最小协议扩展（不破坏现有注入式设计）：
   - `capabilities() -> dict`：如 `{"languages": [...], "clone": bool, "diarize": bool, "device": [...]}`。
   - `is_available() -> tuple[bool, str]`：返回可用性与原因（参考 VoiceStudio 的探测模式）。
2. 让现有 `MlxAudioAsrClient`、`PiperSdkTtsClient`、方向一新增的 client 实现上述方法。
3. 在 `backend/app/main.py` 引入按配置选择后端的工厂函数（ASR、TTS 各一），集中构建逻辑。
4. 在健康检查/状态 API（参考 `backend/app/health.py`、`backend/app/api/`）暴露各引擎可用性与能力。
5. 补测试：能力声明的契约测试 + 不可用后端的降级路径测试。

### 验收标准
- 通过配置在 Piper 与 MLX-Audio TTS 之间切换，无需改动业务代码。
- 后端不可用时 API 返回明确原因，而非静默失败。
- 现有全部后端测试保持通过。

---

## 方向三（P2）：会议内联说话人分离（FunASR/SenseVoice）

### 目标
评估并引入 FunASR/SenseVoice 作为会议 ASR 的可选后端，利用 cam++ 内联分离在长会议场景下
获得更省算力的「转写+分离」一体化路径，与现有 Qwen3-ASR + 独立分离阶段并存。

### 步骤
1. 依托方向二的引擎抽象，新增 `FunAsrClient`（`backend/app/models/`），输出对齐
   `workflow.Segment`（`start`/`end`/`text`/`speaker`），复用 VoiceStudio 中
   `_normalize_funasr` 的思路：优先 `sentence_info`（毫秒时间戳 + `spk`），清洗 `<|...|>` 富标签。
2. 在 `backend/app/core/workflow.py` 支持「ASR 已内联分离」的分支：当后端自带 speaker 时，
   跳过或弱化独立说话人分离阶段（保持开发指引要求的时间戳与说话人信息）。
3. 会议 ASR 仅在会议场景启用，不复用普通对话的热词配置（遵循 `development-guide.md` 边界）。
4. 与现有声纹注册（`speaker_registry`）对接：内联 `Speaker N` 标签映射到已注册声纹。
5. 三段录音回归（`scripts/meeting/download_and_test_workflow.py`）对比 Qwen3-ASR 与 FunASR 的
   准确率、说话人一致性与耗时。

### 验收标准
- FunASR 后端在会议 workflow 中产出带时间戳与说话人的分段。
- 三段录音回归中，FunASR 路径的说话人切分与 Qwen3-ASR + 独立分离结果可对比（记录指标）。
- 未启用 FunASR 时，现有会议流程与结果不变。

---

## 边界与非目标

- 不引入 VoiceStudio 的 Electron 桌面端、dubbing 流水线或大而全引擎目录——与 Astra 聚焦定位冲突。
- 不改动会议 workflow 已打磨的阶段边界与协议注入设计；新增引擎一律实现现有契约后接入。
- 不提交模型权重、真实录音与会议产物（沿用 `.gitignore` 与 `~/Astra/` 产物目录约定）。

## 通用验证清单（每个方向提交前）

```bash
PYTHONPATH=backend .venv/bin/pytest -q backend/tests
```

涉及会议脚本时：

```bash
.venv/bin/python -m py_compile scripts/meeting/download_and_test_workflow.py scripts/meeting/generate_llm_minutes.py
```

静态测试不能证明真实 MLX 推理、远端 LLM 连通性或完整会议任务成功；需在具备模型和服务的
Mac mini 上单独做验收。
