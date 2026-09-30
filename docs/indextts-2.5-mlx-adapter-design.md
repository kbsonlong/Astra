# Astra IndexTTS 2.5 MLX 适配器设计与实现边界

更新时间：2026-10-01

本文记录实现设计和当前运行边界。目标是在 Apple Silicon 上把第三方
`index-tts-2.5-mlx` 作为可选 TTS 后端接入 Astra，同时为后续 TTS 音色库保留稳定的
`voice_id`、授权和模型版本边界。

## 1. 设计结论

- 新增后端名：`indextts_mlx`。
- 首期采用 Astra 进程内适配器 `IndexTTS25MlxClient`，不改造现有 `MlxAudioTtsClient`。
- `index-tts-2.5-mlx` 保持可选依赖，不进入默认 `requirements.txt`；未选择该后端时不导入它。
- 模型权重必须由预下载步骤准备；服务请求默认禁止自动下载，避免首个对话阻塞或意外耗尽磁盘。
- 模型和 MLX 对象只在一个专用 worker 线程中创建和调用；同一个客户端实例串行合成。
- 当前保持现有 `async synthesize(text) -> WAV bytes` 契约，同时支持
  `synthesize_snapshot(text, snapshot)` 的会话级音色选择。
- WebSocket `start_session` 接收 `voice_id` 和可选 `voice_revision`，服务端在会话开始时解析并冻结
  `TtsVoiceSnapshot`，后续每轮生成复用该快照，不复用会议 `SpeakerProfileStore` 的识别语义。
- GGUF 不纳入本适配器；它应是未来独立的 `audio.cpp/Rodel` 后端。

## 2. 已核对的第三方 API 边界

当前本地包检查得到 `index-tts-2.5-mlx==0.1.1`，其 Python 入口为：

```python
from index_tts_2_5_mlx import IndexTTS

tts = IndexTTS(
    model_dir="/absolute/model-dir",
    use_normalization=True,
)
speaker = tts.build_speaker("/absolute/reference.wav")
pcm = tts.synthesize(
    "你好，Astra。",
    lang="zh",
    spk=speaker,
    greedy=True,
)
```

约束：

- `IndexTTS` 初始化会加载完整 pipeline；不能在 FastAPI 事件循环或普通请求线程中直接初始化。
- `build_speaker()` 会从参考音频生成内存中的 `SpeakerContext`，适合按音色版本缓存。
- `synthesize()` 返回 int16 PCM 数组，采样率为 `IndexTTS.sample_rate`，当前包为 22050Hz。
- `clone()` 只是 `synthesize()` 加 WAV 写文件的便捷方法；Astra 应直接拿 PCM 编码到内存，避免中间文件。
- CLI 的参考音频约束为不超过 15 秒；Astra 音色入库应收紧为 5–15 秒、单人、清晰、授权的 WAV。
- 上述是包级 API 检查结果，不等于目标 Mac 已完成真实模型推理；当前 L2 仍未通过。

## 3. Astra 现有边界与改动点

当前代码已经提供可复用的基础：

| 位置 | 当前职责 | 适配器接入方式 |
| --- | --- | --- |
| `backend/app/models/tts_client.py` | Piper、MLX Audio、CosyVoice client | 新增独立 `IndexTTS25MlxClient` |
| `backend/app/config.py` | 环境变量和 `tts_backend` 白名单 | 增加 `indextts_mlx` 及专属配置 |
| `backend/app/main.py` | `_build_tts_client()` 工厂、生命周期 | 增加分支；继续由 lifespan 调用 `aclose()` |
| `backend/app/health.py` | `capabilities()` / `is_available()` | 暴露后端能力和非敏感原因 |
| `backend/app/core/pipeline.py` | 异步调用 TTS、封装 WAV 事件 | 选择性调用 `synthesize_snapshot` |
| `backend/app/api/ws_session.py` | generation 过滤和打断 | 会话启动解析 voice snapshot，不让旧音频跨 generation 播放 |
| `backend/app/core/speaker_registry.py` | 会议说话人识别、embedding、审核 | 不直接复用为 TTS 音色库 |

## 4. 适配器职责

### 4.1 对外契约

适配器必须实现现有 client 的最小契约：

```python
async def synthesize(self, text: str) -> bytes: ...
def is_ready(self) -> bool: ...
def is_available(self) -> tuple[bool, str]: ...
def capabilities(self) -> dict[str, object]: ...
async def aclose(self) -> None: ...
```

返回值必须是单声道、16-bit PCM WAV bytes。`VoicePipeline` 会从 WAV 读取实际采样率，
不能把 22050Hz 写死在 WebSocket 层。

### 4.2 构造参数

建议构造器使用以下参数，不把 token 或真实路径写进日志和健康响应：

```python
IndexTTS25MlxClient(
    model_dir: str,
    repo_id: str = "yunfengwang/IndexTTS-2.5-mlx",
    reference_wav: str = "",
    default_lang: str = "zh",
    use_normalization: bool = True,
    allow_download: bool = False,
    model_revision: str = "",
    model_instance: object | None = None,       # 测试注入
    model_loader: Callable[..., object] | None = None,  # 测试注入
)
```

`model_instance` 和 `model_loader` 只用于单元测试，生产路径必须在 worker 线程中加载。
`model_dir` 优先于 `repo_id`；当 `allow_download=false` 且目录不存在时，健康检查应返回
明确的未就绪原因，而不是在第一次合成时隐式访问 Hugging Face。

### 4.3 内部生命周期

```text
IndexTTS25MlxClient.__init__
        |
        +-- 保存配置，不加载模型、不下载权重
        |
        +-- ThreadPoolExecutor(max_workers=1)
                    |
                    +-- 首次请求：加载 IndexTTS
                    +-- build_speaker(reference hash) -> SpeakerContext
                    +-- synthesize(text, spk, generation options)
                    +-- PCM -> WAV bytes
        |
        +-- aclose(): 拒绝新请求，等待 worker，释放 executor
```

模型实例、`SpeakerContext` 和 MLX 数组都必须由同一 worker 线程创建和使用。不要把
`IndexTTS` 放入当前 `MlxAudioTtsClient` 的全局模型缓存：IndexTTS 包含多个模型组件，
适配器自己的 executor 和实例生命周期更容易保证线程亲和性及关闭行为。

### 4.4 音色上下文缓存

首期默认参考音频可以在客户端内缓存一个 `SpeakerContext`。引入音色库后，缓存键必须是：

```text
(voice_id, voice_revision, reference_sha256)
```

不能只按路径缓存。文件被替换、音色被归档或授权被撤销时，旧上下文必须失效；缓存只存在
于进程内，不作为可持久化模型数据。

建议按请求执行以下检查：

1. `voice_id` 存在且状态为 `active`；
2. `consent_status == confirmed`；
3. 当前参考音频 SHA 与音色记录一致；
4. 参考音频时长在 5–15 秒，单声道且可解码；
5. 取缓存的 `SpeakerContext`，未命中则在同一个 worker 中调用 `build_speaker()`；
6. 将 `voice_revision`、SHA 和模型 revision 写入任务快照。

## 5. 参数映射

### 5.1 Astra 到 IndexTTS

| Astra 参数 | IndexTTS 参数 | 默认策略 |
| --- | --- | --- |
| `text` | `text` | 必须非空，沿用现有句子切分 |
| `language` | `lang` | `zh`；只允许包已验证的语言集合 |
| `reference_wav` | `ref_audio_path` 或 `spk` | 首期默认音色；后续优先 `spk` |
| `greedy` | `greedy` | 默认 false，质量回归后再固定 |
| `seed` | `seed` | 可选，任务重放时保存 |
| `temperature` | `temperature` | 可选，不写入全局配置 |
| `duration_factor` | `duration_factor` | 可选，映射 Astra 语速控制 |
| `n_timesteps` | `n_timesteps` | 先使用上游默认值 |
| `cfg_rate` | `cfg_rate` | 先使用上游默认值 |
| `max_text_tokens_per_segment` | 同名参数 | 先使用上游默认值 |

首期只开放 `lang`、`greedy`、`seed`、`duration_factor` 四项；其余参数等 L2/L4 质量结果
稳定后再暴露，避免把第三方实验参数直接变成 Astra 公共协议。

### 5.2 兼容现有 VoicePipeline

`VoicePipeline._synthesize()` 在没有快照时继续调用旧的 `synthesize(text)`；有快照时通过能力检测
调用 `synthesize_snapshot(text, snapshot)`。这样 Piper、MLX Audio 和现有 CosyVoice 的旧路径不变。

会话启动消息使用以下字段：

```json
{
  "type": "start_session",
  "voice_id": "voice-...",
  "voice_revision": 3
}
```

只有支持该能力的 client 才能处理音色快照；不支持的后端会返回明确的 pipeline 错误，避免静默回退到
错误音色。

## 6. 配置设计

建议在 `Settings` 增加以下字段，默认不改变现有 Piper 行为：

```text
TTS_BACKEND=indextts_mlx
TTS_INDEXTTS_MODEL_DIR=~/.astra/models/indextts-2.5-mlx
TTS_INDEXTTS_REPO_ID=yunfengwang/IndexTTS-2.5-mlx
TTS_INDEXTTS_MODEL_REVISION=<pinned revision>
TTS_INDEXTTS_USE_NORMALIZATION=true
TTS_INDEXTTS_ALLOW_DOWNLOAD=false
```

规则：

- `TTS_BACKEND` 白名单加入 `indextts_mlx`，未知语言和非正数配置启动即拒绝。
- `TTS_INDEXTTS_MODEL_DIR`、token 不得由 `/api/config` 回显。
- `/api/config` 只返回 `tts_indextts_model_configured`、`tts_indextts_model_revision`、
  `tts_indextts_allow_download` 和非敏感能力字段；参考音频由已授权的 voice snapshot 选择。
- `allow_download` 默认 false。下载应由独立安装/准备命令完成，并记录 revision 和 checksum。
- 配置热更新首期不重建 TTS client；模型、参考音色和 executor 必须通过显式重载或进程重启
  切换，避免旧请求引用已撤销的上下文。

## 7. 健康检查与错误语义

### 7.1 `capabilities()` 建议结果

```json
{
  "engine": "indextts25_mlx",
  "backend": "indextts_mlx",
  "model": "IndexTTS-2.5-mlx",
  "model_revision": "<revision>",
  "languages": ["zh", "en", "ja", "yue"],
  "clone": true,
  "clone_configured": true,
  "streaming": false,
  "sample_rate": 22050,
  "device": ["mps"],
  "download_allowed": false
}
```

`model_revision` 可以回显，真实路径和 token 不可以回显。

### 7.2 `is_available()` 分层

健康检查不得触发模型下载或完整加载，按以下顺序返回第一个失败原因：

1. 后端配置是否为 `indextts_mlx`；
2. Python 包是否可导入；
3. 模型目录是否存在且包含预期文件；
4. 默认参考音频是否存在、可读且通过 5–15 秒校验；
5. revision 是否匹配配置（若配置了固定 revision）。

模型尚未在 worker 中加载时可以报告 `available=true, reason=ready (lazy)`；模型首次加载
或 Metal 内存错误在合成请求中转换为稳定的 `TTSClientError`，并记录不含本地路径的日志。

### 7.3 错误分类

建议内部使用稳定错误码，HTTP/WebSocket 层继续沿用现有 `pipeline_failed` 外壳：

| 错误码 | 触发条件 | 是否可重试 |
| --- | --- | --- |
| `indextts_not_installed` | 包未安装 | 否，运维修复 |
| `indextts_model_unavailable` | 权重缺失或 revision 不符 | 否，准备模型 |
| `indextts_reference_invalid` | 音色不存在、未授权、音频不合规 | 否，修正音色 |
| `indextts_busy` | 队列或超时策略拒绝请求 | 是，退避后重试 |
| `indextts_inference_failed` | Metal/模型推理异常 | 一次恢复重试，随后熔断 |
| `indextts_closed` | 应用正在退出 | 否 |

错误日志可包含后端、模型 revision、voice revision 和耗时，但不能包含 token、完整参考路径
或参考文本。

## 8. 测试设计

### 8.1 单元测试，不需要真实模型

在 `backend/tests/test_tts_client.py` 增加注入 fake model：

1. 空文本拒绝；
2. fake `IndexTTS.synthesize()` 返回 int16 PCM，适配器生成 22050Hz 单声道 WAV；
3. `build_speaker()` 对同一 `(voice_id, revision, sha256)` 只调用一次；
4. 不同 revision 或 SHA 不复用旧 `SpeakerContext`；
5. 模型加载只发生一次；
6. 加载和合成运行在专用线程，事件循环不被阻塞；
7. 推理异常转成稳定的 `TTSClientError`；
8. `aclose()` 后拒绝新请求并等待 executor 退出；
9. 未安装包、模型目录缺失、参考音频超时在 `is_available()` 返回可读原因；
10. 真实路径不会出现在 capabilities、health 或错误消息中。

### 8.2 工厂、配置和健康检查

1. `TTS_BACKEND=indextts_mlx` 选择 `IndexTTS25MlxClient`；
2. 默认 `piper` 不触发 IndexTTS import；
3. `TTS_INDEXTTS_DEFAULT_LANG` 非法时 `Settings.from_env()` 失败；
4. `/api/config` 只返回 configured/revision，不返回路径；
5. `/api/health` 返回 `mode=indextts25_mlx`、能力和非敏感 reason；
6. app lifespan 会调用 adapter 的 `aclose()`。

### 8.3 真机验收

依赖独立的 L2/L3 验证：

- 首次加载和短句合成成功；
- 同一进程连续 3 次使用同一 `SpeakerContext`；
- 参考音色切换不会串音；
- 中文、数字、英文混合和长句的音质可接受；
- 冷/热启动耗时、RTF、统一内存和失败恢复有记录；
- WebSocket generation 打断、Piper 回退和 10 轮连续对话回归通过。

在 L2 未通过前，不能把 `indextts_mlx` 设为 Astra 默认后端。

## 9. 分阶段实施顺序

### M0：适配器最小实现

- 已新增 client、配置白名单和工厂分支；
- 已保留可选依赖懒加载；
- fake model 单元测试、health 测试和 `TTS_BACKEND=piper` 全量回归已通过。

### M1：目标 Mac 独立真机验证

- 预下载并固定 model revision；
- 使用授权且 5–15 秒的 WAV 完成 L2/L3；
- 记录内存、RTF、冷/热启动和异常恢复；
- 只有通过后才进入 Astra WebSocket 集成验收。

### M2：TTS 音色库和请求上下文

- 已新增独立 `tts_voices`/revision 表和 API，详见
  [`TTS 音色库数据库与 API 设计`](tts-voice-library-design.md)；
- 已保存授权状态、参考 SHA 和 voice revision；模型 revision 由运行配置提供；
- WebSocket 会话已携带 `voice_id` 和可选 `voice_revision`，并冻结 `TtsVoiceSnapshot`；
- 归档/撤销不会改变已完成任务引用的历史 revision；
- 前端音色选择、任务持久化元数据和真实模型 L6 仍待完成。

### M3：可选 sidecar

仅当第三方依赖与 Astra 主环境冲突、模型加载导致进程不稳定，或需要独立重启/限流时，
再把同一协议移到 localhost/Unix socket sidecar。sidecar 不改变音色、授权和任务快照语义。

## 10. 完成标准

适配器设计只有在以下条件全部满足后才算可进入实现：

- 首期接口不破坏 Piper、MLX Audio、CosyVoice 现有 client；
- 不选择 IndexTTS 时不导入其依赖、不下载权重；
- 请求路径不会隐式联网下载；
- 参考音频授权、长度、SHA 和 revision 可追溯；
- 模型与 speaker context 在同一 worker 线程复用；
- WAV 采样率来自模型实际输出；
- health、错误和日志不泄露本地路径或 token；
- 有 fake model 单测、工厂/配置/健康测试和真机 L2/L3 验收记录；
- L2/L3 未通过时，后端保持 opt-in，不作为默认 TTS。
