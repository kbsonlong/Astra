# Qwen3-ASR 会议领域微调接入设计

> 状态：设计与配置接入
> 更新：2026-09-09
> 适用模型：`Qwen3-ASR-0.6B` 训练基座、`mlx-community/Qwen3-ASR-0.6B-4bit` 会议推理模型

## 1. 目标与结论

Astra 的首要目标是提高每周周会逐字稿的可靠性，尤其是：

- 专业词汇、产品名、英文缩写、版本号和数字；
- 参会人员的发言内容与说话人归属；
- 在此基础上生成不臆造事实的会议纪要。

关键结论：

1. 会议纪要不能直接作为 ASR 训练标签。ASR 训练必须使用音频与逐字文本一一对应的 `audio/text` 样本；纪要是摘要，不能替代原话。
2. “听清内容”和“判断是谁在说话”是两个任务。Qwen3-ASR 负责语音转文字，Resemblyzer/后续说话人模型负责声纹聚类和人员匹配。
3. 原始逐字稿和人工确认稿必须双轨保存。只有人工确认的 `confirmed transcript` 才能用于纪要和训练，不能覆盖 `raw transcript`。
4. Mac mini 可以完成模型推理、音频预处理、数据审校和评估；官方 Qwen3-ASR 微调流程是 PyTorch/`torchrun`，实际生产训练应使用 NVIDIA CUDA 环境。
5. CUDA 不是神经网络训练在理论上的绝对必要条件，但 Apple Silicon MPS 不是 Qwen3-ASR 官方微调路径。当前项目不能把 Mac mini 直接微调 4bit MLX 权重视为受支持方案。

## 2. Astra 当前链路与接入位置

当前会议链路已经具备较好的基础：

```text
POST /api/meeting/process
  -> 独立 meeting worker
  -> 音频解码为 16kHz mono WAV
  -> Silero VAD
  -> Qwen3-ASR 逐段转写
  -> 标点恢复
  -> Resemblyzer 声纹分离和已注册人员匹配
  -> 确定性/受限候选纠错
  -> 远端 LLM 生成纪要
```

相关代码边界：

- `backend/app/api/meeting_routes.py`：创建任务、保存输入和订阅状态；
- `backend/app/core/meeting_cli.py`：独立进程执行长会议；
- `backend/app/core/workflow.py`：VAD、ASR、标点、说话人分离和纠错编排；
- `backend/app/models/asr_client.py`：MLX ASR 适配器；
- `backend/app/core/speaker_registry.py`：人员档案和声纹样本；
- `scripts/meeting/enroll_speakers_from_meetings.py`：从已有会议片段初始化声纹；
- `frontend/src/UploadPage.tsx`：上传、会议处理和声纹人工审核界面。

训练功能不应放入 FastAPI 请求进程。前端只负责训练参数展示和保存，实际训练由独立脚本或外部 GPU 机器执行。

## 3. 数据闭环

### 3.1 会议处理后的双轨数据

每个会议段建议保存为 JSONL，而不是只保存一份纯文本：

```json
{
  "segment_id": "seg-000123",
  "start": 125.42,
  "end": 131.08,
  "speaker_id": "speaker-uuid",
  "speaker_name": "张三",
  "speaker_confidence": "high",
  "raw_text": "冷资源中心这块本周要继续看",
  "corrected_text": "云资源中心这块本周要继续看",
  "correction_source": "human",
  "review_status": "approved",
  "term_candidates": ["云资源中心"],
  "asr_model": "qwen3-asr-0.6b-4bit-baseline"
}
```

要求：

- `raw_text` 永远只读保存；
- `corrected_text` 只允许人工确认或受限规则产生；
- 低置信度说话人标记为待确认，不自动写入真实姓名；
- 重叠讲话、背景噪声、无法辨认片段应标记出来，不要猜测；
- 会议纪要只读取已确认文本；
- 训练数据只读取 `review_status=approved` 的片段。

### 3.2 Qwen3-ASR 训练 JSONL

官方微调格式是音频路径和文本字段，例如：

```json
{"audio":"/data/asr/clips/meeting-001/seg-000123.wav","text":"language Chinese<asr_text>本周先看云资源中心的成本优化。"}
```

训练文本必须是音频中实际说出的内容，不应包含：

- 会议摘要；
- LLM 推断出的结论；
- `S1`、`张三` 等说话人标签；
- 人工不确定但未经确认的专有名词；
- 为了“读起来通顺”而改写的句子。

建议统一文本规范：专业名词、英文和数字保持稳定写法；如果标点由后置模型负责，训练集要保持一致的标点策略。

会议点击“生成会议纪要”完成后，worker 会在该任务目录生成以下训练候选产物：

```text
<meeting_output_dir>/<task_id>/
  asr_clips/seg-000001.wav
  transcript_segments.jsonl       # 时间戳、说话人、raw/corrected 双轨详情
  qwen3-asr-candidates.jsonl      # audio/text 训练候选，默认 review_status=pending
```

`qwen3-asr-candidates.jsonl` 已符合 Qwen3-ASR 的 `audio`/`text` 输入结构，但候选样本必须经过人工审校并改为 `review_status=approved` 后才能训练。训练脚本会跳过 `pending` 样本；没有 `review_status` 字段的旧版人工数据仍兼容读取。

### 3.3 数据集拆分

必须按会议拆分，而不是随机按片段拆分：

```text
train：会议 A、B、C、D
 dev ：会议 E
 test：会议 F
```

同一场会议的片段不能同时出现在训练集和测试集，否则会高估模型效果。初始建议先精校 5～10 场会议，形成 2～5 小时 Gold 数据；正式微调前再逐步积累到 10 小时以上，并覆盖不同房间、麦克风距离、噪声、参会人数、中英混说和多人抢话情况。

真实音频、人员姓名和会议内容放在 `~/Astra/data/` 等本机目录，不提交到 Git。

## 4. 术语与说话人策略

### 4.1 术语

术语应当独立维护 canonical name 和 aliases：

```yaml
terms:
  - canonical: 云资源中心
    aliases: ["冷资源中心", "云资源中心"]
  - canonical: host 网络模式
    aliases: ["后视网络模式", "host网络模式"]
```

术语词典用于候选召回、人工审校和评估，不应对所有文本做无条件替换。

Qwen3-ASR 支持 context biasing，但 Astra 当前已观察到长 system prompt/hotwords 可能被模型复读到输出末尾。因此会议主链路继续使用干净 ASR：

1. 第一遍不注入长热词表，保留原始转写；
2. 只对疑似术语的短片段做候选二次识别；
3. 只有通过约束校验或人工确认，才进入 `confirmed transcript`。

### 4.2 说话人

说话人识别独立于 ASR：

```text
音频片段 -> 声纹 embedding -> 聚类 -> speaker_id -> display_name
```

每位参会人员建议注册 3～5 段不同语气和距离的样本。未知人员进入 `pending_review`，人工播放样本、改名并审核后，才参与后续会议匹配。不能使用 ASR 微调来替代说话人分离。

## 5. 训练参数设计

Astra 新增独立训练配置文件，默认路径为：

```text
~/.astra/qwen3-asr-training.json
```

后端接口：

```text
GET /api/training/config
PUT /api/training/config
```

前端“音频工作台”中的“领域微调参数”面板可以展示和修改以下参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `model_path` | `Qwen/Qwen3-ASR-0.6B` | 训练基座，建议使用非量化 checkpoint |
| `train_file` | `~/Astra/data/asr/train.jsonl` | 训练数据 |
| `eval_file` | `~/Astra/data/asr/dev.jsonl` | 验证数据 |
| `output_dir` | `~/Astra/models/qwen3-asr-meeting` | checkpoint 输出目录 |
| `device` | `cuda` | `cuda`、`mps`、`cpu` 或 `auto` |
| `precision` | `bf16` | `bf16`、`fp16` 或 `fp32` |
| `batch_size` | `1` | 单卡显存不足时优先保持为 1 |
| `grad_acc` | `8` | 梯度累积，等效扩大 batch |
| `learning_rate` | `2e-5` | 官方示例学习率 |
| `epochs` | `1` | 领域小数据先从 1 开始 |
| `save_steps` | `200` | checkpoint 保存间隔 |
| `save_total_limit` | `3` | 最多保留 checkpoint 数 |
| `num_workers` | `2` | 数据加载线程 |
| `pin_memory` | `true` | CUDA 数据传输优化项 |
| `persistent_workers` | `true` | 保持数据加载线程 |
| `prefetch_factor` | `2` | 数据预取数量 |
| `resume_from` | 空 | 指定恢复的 checkpoint |
| `resume_latest` | `false` | 是否自动恢复输出目录中的最新 checkpoint |

前端保存配置的语义是“保存离线训练参数”，不是启动训练，不会热切换当前运行模型。训练脚本应读取该 JSON 并显式打印最终参数，避免配置和实际命令不一致。

## 6. Mac mini 是否可以完成微调

### 6.1 可以完成的工作

Mac mini 适合完成：

- 会议录音采集和原始文件保存；
- VAD、ASR、标点和声纹推理；
- 人工审校和术语确认；
- 导出 `audio/text JSONL`；
- 使用基础模型和候选模型跑固定测试集；
- 生成 CER、术语召回率和说话人指标；
- 运行转换后的 MLX 推理模型。

### 6.2 当前可选的训练路径

`mlx-tune` 已提供基于 MLX 的 Qwen3-ASR LoRA 微调实现，可以在 Apple Silicon 上训练音频编码器和 Qwen3 解码器；当前公开示例使用 `mlx-community/Qwen3-ASR-1.7B-8bit`。因此 Mac mini 可以作为实验和小规模领域适配环境，但 `Qwen3-ASR-0.6B-4bit` 仍需在本机先做加载、反向传播和导出验证，不能仅凭推理成功认定训练链路可用。

正式训练仍建议保留 CUDA 路径，原因是：

1. 官方微调代码以 PyTorch 和 `torchrun` 为主，推荐 CUDA/FlashAttention 2；
2. `mlx-community/Qwen3-ASR-0.6B-4bit` 是量化权重，是否能在当前 `mlx-tune` 版本上直接挂载 LoRA，需要以实际 smoke test 为准；
3. 训练需要梯度、优化器状态和激活内存，远高于 4bit 推理；
4. Apple MPS 可能能运行部分普通 PyTorch 算子，但不等于官方 Qwen3-ASR SFT 脚本可运行；
5. 0.6B ASR 不只有文本 decoder，还包含音频 encoder、projector、数据整理和长音频处理；
6. 当前项目尚未完成 0.6B-4bit 在 Apple Silicon 上的断点恢复和精度回归验证。

因此，16GB Mac mini 上可以先完成小规模 LoRA 实验，但在通过固定会议测试集、长音频和断点恢复验证前，不把它作为唯一生产训练方案。Mac mini 负责数据和评估，NVIDIA GPU 机器作为稳定训练回退路径。

### 6.3 是否“一定”依赖 CUDA

分三种情况理解：

| 场景 | 是否建议 | 结论 |
|---|---|---|
| 官方 Qwen3-ASR SFT | 是 | 以 CUDA/PyTorch/`torchrun` 作为可复现主路径 |
| Apple MPS 自定义移植 | 否 | 理论上可能，但需要自行适配、速度和算子兼容性不保证 |
| CPU 训练 | 否 | 理论可运行某些算子，但速度和内存占用不适合实际迭代 |

准确表述是：**CUDA 不是数学意义上的绝对必要条件，但对当前官方 Qwen3-ASR 微调流程而言，CUDA 是应当依赖的生产环境。Mac mini 不需要 CUDA 来做推理，但不能把 MPS 微调当成已验证能力。**

如果要在 Mac mini 上把 `Qwen3-ASR-0.6B-4bit` 作为正式训练路径，需要单独完成：

- 确认 `mlx-tune` 对目标 4bit checkpoint 的加载、LoRA 注入和保存行为；
- 确认量化权重上的 LoRA 训练不会破坏推理导出；必要时切换到 8bit 或 bf16 基座；
- 确认 audio encoder、projector、decoder 的算子均支持 MPS；
- 使用短音频、小 batch、梯度累积和 checkpoint 断点恢复；
- 对比 CPU/MPS/CUDA 的 loss、速度和验证集 CER；
- 证明导出的 MLX 模型在 Astra 上没有精度回退。

在这些条件完成前，前端中的 `mps` 选项只代表实验配置，不代表 Astra 已支持在 Mac mini 上正式训练。

## 7. 评估和模型发布门禁

每个候选模型都必须与当前 `Qwen3-ASR-0.6B-4bit` 基线在同一测试集上比较：

### ASR

- 中文 CER；
- 英文 WER；
- 人名准确率；
- 专业术语召回率和精确率；
- 数字、金额、版本号准确率；
- 中英混说片段错误率；
- 重叠讲话和噪声片段错误率。

### 说话人

- 说话人 DER；
- 已注册人员识别准确率；
- 未知人员误匹配率；
- 低置信度是否进入待审核。

### 发布规则

- 候选模型不能在通用测试集和会议测试集上出现明显回退；
- 术语和人名指标必须单独达标，不能被整体 CER 掩盖；
- 不能出现 prompt 复读、大段 hallucination 或数字异常改写；
- 只有通过测试集门禁后，才切换 `MEETING_ASR_MODEL`；
- 保留上一版本模型和配置，支持一键回滚。

建议每周只做数据审校；积累足够 Gold 数据后再训练候选模型，不要每上传一场会议就自动改写生产模型。

## 8. 推荐落地阶段

### Phase 1：数据和审校闭环

- 保存 raw/confirmed 双轨文本；
- 在前端增加逐句审校、术语确认和说话人审核；
- 纪要只使用 confirmed transcript；
- 建立固定 Golden Test Set。

### Phase 2：术语候选和声纹增强

- 引入团队术语词典；
- 对疑似术语片段做二次识别；
- 完善每位参会人员的声纹样本；
- 统计高频错误，不立即训练。

### Phase 3：CUDA 环境领域微调

- 导出 approved 的 `audio/text JSONL`；
- 用官方 PyTorch 训练脚本在 CUDA 机器训练；
- 先做 1 epoch 小实验；
- 评估后转换为 MLX；
- Mac mini 上跑完整回归；
- 通过门禁后灰度上线。

### Phase 4：版本化迭代

```text
每周：采集和审校
每月或数据足够后：训练候选模型
候选通过固定测试集：灰度使用
连续稳定：正式切换
出现回退：恢复上一模型
```

## 9. 参考资料

- [Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR 官方 Fine-tuning README](https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md)
- [Qwen3-ASR Technical Report](https://arxiv.org/abs/2601.21337)
- Astra 当前会议链路：`backend/app/core/workflow.py`、`backend/app/core/meeting_cli.py`
