# Astra 语音增强模型栈设计

> 状态：Phase B 已接入 ZipEnhancer 离线链路；模型默认关闭，尚未设为生产默认
>
> 日期：2026-09-11
>
> 目标：把背景噪声、回声和多人重叠说话纳入 Astra 的音频前处理选择，而不是把七个模型无条件串联。

## 1. 背景与结论

Astra 当前的主要链路是：

```text
实时会话：浏览器音频 -> WebSocket -> ASR -> LLM -> TTS
会议上传：音频文件 -> afconvert/16 kHz mono -> VAD -> ASR -> 标点 -> 声纹/说话人 -> 纪要
```

这条链路已经具备 VAD、ASR 和说话人处理，但“输入给 ASR 的声音是否干净”仍主要依赖原始录音。新增模型应作为可选的 `AudioEnhancementStage` 插入，而不是替换 ASR 或 VAD：

```text
输入采集
  -> [可选 AEC：需要远端参考]
  -> [可选 ANS：背景噪声抑制]
  -> [按需 Separation：多人重叠时拆分]
  -> VAD
  -> ASR
  -> 标点 / 说话人 / 纠错 / 纪要
```

核心决策：

1. AEC 只在有“麦克风信号 + 播放给远端/扬声器的参考信号”时启用；单个上传文件没有参考信号，不能假装做回声消除。
2. ANS 每条链路最多选择一个主模型。ZipEnhancer、DFSMN ANS、FRCRN 不是默认全部串联，否则容易过度抑制、增加延迟并难以定位音质损失。
3. Separation 是昂贵的按需分支。先检测重叠说话或由用户选择“多人分离”，再将独立流送入 VAD/ASR；普通单人会议不默认运行。
4. 16 kHz mono PCM 是 ASR 的规范化边界；8 kHz/48 kHz 模型通过显式重采样适配，禁止让每个模型自行隐式重采样。
5. 原始输入永远保留，增强结果、模型、采样率、延迟和回退原因写入任务元数据，便于复听、审校和回归比较。

## 2. 七个模型的能力矩阵

| 模型 | 能力 | 采样率 | 输入/输出约束 | 推荐角色 | 主要风险 |
|---|---|---:|---|---|---|
| [JAEC 16K](https://modelscope.cn/models/iic/speech_jaec_aec_16k) | 可解释声学回声消除 | 16 kHz | 麦克风信号 + 远端参考；TDE + LP | 实时通话/会议前端第一阶段 | 当前公开版本主要处理线性回声，无法覆盖扬声器破音等非线性失真 |
| [DFSMN AEC 16K](https://modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k) | 实时回声消除 | 16 kHz | 需要按模型卡提供的参考信号 | JAEC 的基线/回退方案 | 可解释性和中间诊断能力较弱，必须实测双讲场景 |
| [ZipEnhancer 16K](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base) | 单麦语音降噪 | 16 kHz | 单路麦克风 | 语音助手、ASR、参考音频清理 | 复杂混响、强非平稳噪声下可能残留或伤害辅音 |
| [FRCRN 16K](https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k) | 复杂噪声语音增强 | 16 kHz | 单路麦克风 | 噪声更复杂的离线会议/录音 | 计算量、延迟和模型占用需在 Apple Silicon 上实测 |
| [DFSMN ANS 48K](https://modelscope.cn/models/iic/speech_dfsmn_ans_psm_48k_causal) | 因果实时降噪 | 48 kHz | 单路 48 kHz 流 | 高采样率实时通话/采集 | 不能直接当作 16 kHz ASR 前端，需明确重采样和带宽代价 |
| [FLASepformer](https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100) | 单通道多人语音分离 | 8 kHz | 混合语音，公开模型面向 2-speaker 数据 | 多人会议/访谈的重叠语音分支 | 公开 checkpoint 的语言、说话人数和领域覆盖有限，不能等同于通用中文多人分离 |
| [MossFormer2 8K](https://modelscope.cn/models/iic/speech_mossformer2_separation_temporal_8k) | 单通道语音分离 | 8 kHz | 混合语音 | Separation 对照组/回退方案 | 与 FLASepformer 的效果、资源和长音频行为需要同场 benchmark |

FLASepformer 的论文将其重点放在长序列语音分离：通过 Gated Focused Linear Attention 把注意力处理从二次复杂度改为线性复杂度。论文报告的 30 秒、RTX A800 实验中，FLA-SepReformer-T/B/L 相对对应基线约为 2.29x/1.91x/1.49x 速度，显存占用约为 15.8%/20.9%/31.9%；这些数字是论文环境结果，不应直接当作 Astra 的 Apple Silicon SLA。[论文](https://arxiv.org/abs/2508.19528)

JAEC 的公开说明将处理链拆成神经 TDE（估计并对齐延迟）和神经 LP（估计并抵消线性回声），16 kHz 版本按 10 ms 帧处理，公开资料报告固定算法延迟为 22 ms，并明确暂不包含非线性处理网络。集成时应把 TDE、LP 输出和回退状态纳入可观测信息，而不是只保存最终波形。[模型发布说明](https://www.nxrte.com/jishu/71864.html)

## 3. 按场景的推荐组合

| 场景 | 默认组合 | 处理顺序 | 说明 |
|---|---|---|---|
| 语音助手与实时识别 | `ZipEnhancer 16K`；48 kHz 输入则评估 `DFSMN ANS 48K` | ANS -> VAD/ASR | 追求识别稳定性和低延迟；不要同时叠加 FRCRN |
| 噪声较复杂的会议上传 | `FRCRN 16K` | FRCRN -> VAD -> ASR | 只对离线会议启用，保留原音供人工比较 |
| 实时通话与会议 | `JAEC 16K` -> `ZipEnhancer 16K` | AEC -> ANS -> VAD/ASR | 只有存在远端播放参考时成立；JAEC 失败时降级到 DFSMN AEC 或原始麦克风 |
| 高采样率实时采集 | `DFSMN ANS 48K` | ANS 48K -> 重采样 16K -> VAD/ASR | 不要先降到 16K 再调用 48K 模型 |
| 多人会议与访谈 | `FLASepformer`；以 `MossFormer2` 作对照 | AEC/轻量 ANS -> Separation -> 每路 VAD/ASR | Separation 只在重叠检测或用户显式选择时运行 |
| 只有单人、无远端参考的会议文件 | 不启用 AEC；选择 ZipEnhancer 或 FRCRN | ANS -> VAD -> ASR | 文件本身没有可供 AEC 对齐的远端参考 |
| 训练数据清理 | ANS/Separation 仅生成候选，原音和增强音并列保存 | 增强 -> VAD/ASR -> 人工审校 | 训练集只消费人工批准的音频/逐字稿对，不自动把增强结果写入训练集 |

### 3.1 为什么不把七个模型串起来

- AEC 和 ANS 都可能改变近端语音的幅度、相位和停顿；重复处理会放大伪影。
- Separation 的输入分布与 ANS 的输入分布不一定兼容，先后顺序必须由 A/B 测试决定。
- 不同采样率之间的往返重采样会增加 CPU 和失真，且会掩盖真正的模型效果。
- 模型出错时需要能定位“是 AEC、降噪、分离还是 ASR”造成的问题；每次只启用明确的组合。

## 4. Astra 集成边界

### 4.1 统一音频对象

建议新增内部对象，不让各模型直接读写临时路径：

```python
@dataclass
class AudioBuffer:
    samples: NDArray[np.float32]
    sample_rate: int
    channels: int
    start_time: float
    source: Literal["microphone", "upload", "reference", "separated"]


@dataclass
class EnhancementContext:
    reference: AudioBuffer | None
    session_id: str
    task_id: str | None
    realtime: bool
    model_config: dict[str, object]
```

阶段接口建议保持窄小：

```python
class AudioEnhancementStage(Protocol):
    name: str
    input_sample_rate: int
    output_sample_rate: int
    realtime: bool

    async def process(
        self, audio: AudioBuffer, context: EnhancementContext
    ) -> tuple[AudioBuffer, EnhancementMetrics]: ...
```

实现要求：

- `AECStage` 显式要求 `context.reference`，没有参考信号时返回 `not_applicable`，不得静默伪造。
- `ANSStage` 只负责降噪，不负责 VAD 或 ASR；输入输出采样率由配置明确。
- `SeparationStage` 返回有序的 `list[AudioBuffer]`，并附带 `source_index`、分离置信度和处理窗口。
- 每个 stage 都可以返回 `applied`、`not_applicable`、`failed` 或 `disabled`，失败时由策略决定回退到上一份音频。
- 所有模型推理必须在受控 worker 中执行，不能在 FastAPI async handler 中直接做同步 PyTorch/ONNX 推理。

### 4.2 插入现有链路的位置

#### 实时 WebSocket

```text
浏览器麦克风帧
  -> 统一采样率/帧长
  -> JAEC（若有 far-end reference）
  -> ANS
  -> Session audio buffer
  -> ASR worker
  -> LLM/TTS
```

实时链路必须以 10 ms 帧为基本单位，维护有界队列。处理超时不能无限堆积：

1. 首选跳过当前增强帧并记录 `enhancement_frame_dropped`；
2. 连续超时达到阈值时关闭增强，继续将原始/最近可用音频送入 ASR；
3. WS 状态事件报告当前增强模式和回退原因，但不把模型内部日志直接发送给客户端。

JAEC 需要远端播放参考帧。前端或播放层必须把实际送往扬声器的 PCM 复制到 `reference` 通道，而不是把网络收到的压缩包或 TTS 请求文本当作参考。

#### `/api/transcribe/stream`

该接口是上传后 SSE，不具备实时 AEC 的远端参考。第一阶段只允许 ANS 和按需 Separation；如果调用方声明没有 reference，就拒绝 `jaec_16k` 配置并返回明确诊断，而不是产生看似成功的结果。

#### 会议工作进程

```text
input.*
  -> 规范化 16K mono / 保留原始 input
  -> 可选 ANS
  -> 可选 FLASepformer/MossFormer2
  -> VAD
  -> ASR
  -> punctuation
  -> speaker diarization / voiceprint candidate
  -> deterministic correction
  -> minutes
```

会议上传没有远端播放参考，因此默认关闭 AEC。Separation 输出应保留为任务产物，供人工复听和排查；ASR 只消费通过 stage policy 选出的分支。

### 4.3 配置建议

首期配置名建议如下，默认值必须保持现有行为：

```dotenv
AUDIO_ENHANCEMENT_ENABLED=false
AUDIO_ANS_MODEL=none                  # none|zipenhancer_16k|frcrn_16k|dfsmn_ans_48k
AUDIO_AEC_MODEL=none                  # none|jaec_16k|dfsmn_aec_16k
AUDIO_SEPARATION_MODEL=none           # none|flasepformer_8k|mossformer2_8k
AUDIO_ENHANCEMENT_DEVICE=auto         # auto|cpu|mps|cuda
AUDIO_ENHANCEMENT_MODEL_DIR=~/.astra/models/audio-enhancement
AUDIO_ENHANCEMENT_MAX_QUEUE=2
AUDIO_ENHANCEMENT_FRAME_TIMEOUT_MS=80
AUDIO_SEPARATION_MAX_SPEAKERS=2
AUDIO_SEPARATION_TRIGGER=manual       # manual|overlap|always
```

配置约束：

- `AUDIO_AEC_MODEL != none` 时，运行时请求必须提供 reference；否则只能在启动检查中标记不可用。
- `AUDIO_ANS_MODEL=dfsmn_ans_48k` 时，输入链路必须声明 48 kHz，不能用 16 kHz 音频直接调用。
- `AUDIO_SEPARATION_TRIGGER=always` 只允许离线任务使用；实时 WebSocket 禁止默认全量分离。
- 模型下载、许可证和校验和必须记录在模型清单中，不能由请求参数直接指定任意远程模型路径。

## 5. 运行资源与部署策略

### 5.1 进程隔离

不要把七个模型同时加载进 Astra API 进程。建议按工作负载分成三个 worker profile：

| profile | 常驻模型 | 用途 | 并发策略 |
|---|---|---|---|
| `realtime-lite` | 一个 ANS + 可选 JAEC | WebSocket 对话/通话 | 单 worker、有界队列、超时回退 |
| `meeting-enhance` | 一个离线 ANS 或 Separation | 会议任务 | 复用现有 meeting 独立进程，按任务配额串行 |
| `benchmark` | 任意单模型 | 离线 A/B 测试 | 不接生产请求 |

Apple Silicon 上要特别避免 MLX ASR、PyTorch/ONNX 增强模型同时抢占 Metal 内存。首期默认使用 CPU 或单一 MPS 增强 worker，直到真实录音 smoke test 证明内存和实时率稳定。

### 5.2 资源预算与降级

- 每路实时音频只保留有限帧队列；队列满时丢增强帧，不复制无界的原始和增强缓冲。
- 会议分离按固定窗口运行，禁止把数小时输入一次性送入分离模型。
- 记录 `real_time_factor`、p50/p95 stage latency、峰值 RSS/Metal memory、队列深度和丢帧数。
- 任一增强 stage 超时或 OOM 时，保留原始音频并进入 `bypass`；不能让会议任务整段失败。
- 输出文件命名建议：`input.*`、`enhanced/ans.wav`、`enhanced/aec.wav`、`separated/source-0.wav`，并在 `meta.json` 标记实际消费的分支。

## 6. 可观测性与数据契约

每个任务的 `meta.json` 或实时 session 状态至少记录：

```json
{
  "enhancement": {
    "enabled": true,
    "stages": [
      {
        "name": "jaec_16k",
        "status": "applied",
        "input_sample_rate": 16000,
        "output_sample_rate": 16000,
        "latency_ms": 22,
        "reference_present": true,
        "fallback_reason": null
      }
    ],
    "raw_audio_preserved": true,
    "selected_branch": "enhanced"
  }
}
```

JAEC 还应尽可能记录 TDE 的估计延迟、LP 回声能量/抵消结果和 reference 是否连续。FLASepformer 应记录输入窗口、输出路数、模型版本和是否发生截断。不能把这些诊断字段混入 ASR 文本或会议纪要。

## 7. 验证方案

### 7.1 离线数据集

建立固定的、不提交真实隐私录音的 benchmark fixture：

- clean speech + 稳态噪声/键盘/风扇/人声背景；
- far-end reference + near-end speech，覆盖单讲和双讲；
- 两人重叠中文语音，覆盖不同音量、说话速度和重叠比例；
- 16 kHz、8 kHz、48 kHz 以及可识别失败的输入格式。

### 7.2 指标

| 目标 | 主指标 | 不能牺牲的指标 |
|---|---|---|
| ANS | DNSMOS/PESQ、SNR improvement | 中文 ASR WER/CER、辅音清晰度 |
| AEC | ERLE、near-end speech distortion、双讲稳定性 | ASR WER/CER、端到端延迟 |
| Separation | SI-SNRi/SDRi、source leakage | 每路 ASR CER、说话人标签稳定性 |
| 系统 | RTF、p95 latency、峰值内存、回退率 | 任务成功率、原音可恢复性 |

### 7.3 通过门槛

模型只有同时满足以下条件才允许成为默认配置：

1. 在目标 Mac mini 上完成真实录音 smoke test；
2. 实时链路稳定低于配置的 frame timeout，连续 5 分钟无无界积压；
3. 增强后 ASR CER 不劣于原始基线，或在目标噪声集上有可解释的收益；
4. AEC 的单讲、双讲、无 reference 和非线性失真都完成对照测试；
5. Separation 对 2 人重叠语音有效，但超出公开模型覆盖范围时明确降级，不宣称通用多人分离；
6. 模型许可证、权重来源、版本和校验和已记录。

## 8. 分阶段实施计划

### Phase A：接口和 benchmark，不改变默认行为

- 新增 `AudioBuffer`、`EnhancementContext`、`AudioEnhancementStage` 和 `EnhancementMetrics`；
- 实现 `PassthroughStage`，将现有 decode/VAD/ASR 接到 stage seam；
- 增加离线 benchmark 脚本和 `meta.json` 字段；
- 默认 `AUDIO_ENHANCEMENT_ENABLED=false`。

### Phase B：先接入一个 16 kHz ANS

- 优先验证 ZipEnhancer 16K，FRCRN 16K 作为复杂噪声对照；
- 只接会议离线 worker，不立即改实时 WebSocket；
- 用同一份原始录音比较原始/增强两路 ASR 和人工听感。

当前实现：

- `backend/app/core/zipenhancer.py` 通过 ModelScope pipeline 延迟加载
  `iic/speech_zipenhancer_ans_multiloss_16k_base`，只接受已规范化的 16 kHz mono 音频；
- `AUDIO_ENHANCEMENT_ENABLED=false`、`AUDIO_ANS_MODEL=none` 保持默认旁路；设置
  `AUDIO_ANS_MODEL=zipenhancer_16k` 后，仅会议独立 worker 调用增强阶段，实时 WebSocket 不接入；
- `meta.json` 与 `status.json` 记录 stage 状态、模型、延迟和回退原因；原始音频仍保留；
- ModelScope 依赖放在 `backend/requirements-audio-enhancement.txt`，不会随基础安装强制引入，且本阶段不自动下载权重；
- 目前已完成 fake backend 契约测试，真实权重、真实录音听感和 ASR A/B 仍需在目标 Mac mini 上验证。

### Phase C：实时 AEC + ANS

- 前端补充远端 PCM reference 的采集/传递契约；
- 接 JAEC 16K，暴露 TDE/LP 诊断信息；
- 以 DFSMN AEC 16K 作为可切换基线，不做静默双 AEC 串联；
- 验证无 reference、双讲、延迟漂移、扬声器非线性失真。

### Phase D：按需 Separation

- 先接 FLASepformer 8K，增加重叠触发和离线窗口；
- 以 MossFormer2 8K 做效果/资源对照；
- 每路输出独立 VAD/ASR，并保持原始混合流；
- 仅将人工确认的分离结果纳入声纹和训练数据流程。

### Phase E：产品化选择器

- 在前端提供“原始/增强/分离”复听和状态；
- 根据输入采样率、是否有 reference、是否检测到重叠和资源预算选择组合；
- 将模型切换、回退和 benchmark 结果纳入运行配置和审计日志。

## 9. 不在本次设计承诺的事项

- 不承诺 FLASepformer 对任意中文、任意说话人数和任意会议混响都有效；
- 不把 JAEC 16K 当作非线性回声消除器；
- 不在没有远端 reference 的上传文件上启用 AEC；
- 不因为模型开源就假设可以直接商用或重新分发权重；
- 不在真实 Mac mini、麦克风权限、远端 LLM 和长录音 smoke test 完成前把新模型设为生产默认。

## 10. 参考资料

- [FLASepformer 论文（arXiv）](https://arxiv.org/abs/2508.19528)
- [FLASepformer ModelScope 模型卡](https://modelscope.cn/models/iic/speech_flatsepreformer_separation_temporal_8k_base_libri2mix100)
- [JAEC 16K ModelScope 模型卡](https://modelscope.cn/models/iic/speech_jaec_aec_16k)
- [七个语音增强模型及公开指标整理](https://www.nxrte.com/jishu/71864.html)
- [FunASR ModelScope 模型列表与采样率/用途说明](https://github.com/modelscope/FunASR/blob/main/model_zoo/modelscope_models.md)
