# Astra 多引擎 TTS 路由设计

> 更新时间：2026-10-01
> 决策：采用多引擎路由
> 关联：[音色库页面方案](tts-voice-library-page-plan.md) · [实现任务清单](tts-voice-library-implementation-plan.md)

## 1. 决策与目标

Astra 采用多引擎 TTS 路由，而不是由单个 `TTS_BACKEND` 决定所有实时会话。它使系统默认声音（首期为 Piper）和已授权的 IndexTTS 音色能够在同一部署中按会话选择，同时保持 `voice_id + voice_revision` 的不可变快照语义。

目标：

- 未选择克隆音色的会话使用稳定默认引擎；
- 已选择音色的会话按 snapshot 的 `backend_family` 路由到兼容引擎；
- 用户明确选择的音色绝不静默降级为另一种声音；
- 一个引擎不可用、排队或失败时，不影响其他引擎和既有会话；
- 所有模型保持懒加载，避免启动即耗尽 Apple Silicon 的统一内存。

非目标：

- 不在一次会话内切换音色或引擎；
- 不对 IndexTTS、CosyVoice 等克隆引擎做逐句静默回退；
- 不把每种引擎的全部实验参数暴露到浏览器；
- 不以启动时加载所有模型来换取首次响应速度。

## 2. 路由规则

```text
start_session
  ├── 未传 voice_id
  │     └── TtsRouter.synthesize() → default_backend（Piper）
  │
  └── 传 voice_id + voice_revision
        └── TtsVoiceStore.snapshot()
              └── snapshot.backend_family
                    └── TtsRouter.synthesize_snapshot()
                          └── compatible engine（例如 IndexTTS MLX）
```

`backend_family` 到引擎键的首期映射：

| 音色 backend_family | 路由引擎键 |
| --- | --- |
| `indextts25_mlx`、`indextts_mlx` | `indextts_mlx` |
| `cosyvoice` | `cosyvoice` |

Piper 与普通 MLX Audio 仅可作为系统默认声音；除非未来建立对应的可授权、版本化音色适配器，否则它们不能接受 `TtsVoiceSnapshot`。

## 3. 回退策略

| 场景 | 系统行为 | 用户可见结果 |
| --- | --- | --- |
| 默认引擎在生成前不可用 | 尝试配置的默认回退引擎 | 显示“已改用系统回退声音” |
| 默认引擎在某轮已输出后失败 | 不跨句切换声音 | 该轮失败，提示重试或重新开始会话 |
| 指定音色的引擎不可用/队列满 | 不回退到 Piper | 开始会话前阻止；运行中显示该音色不可用 |
| 指定音色 synthesis 失败 | 不回退到另一音色 | 该轮失败，保留音色和错误摘要 |
| 音色授权被撤销 | 阻止新的会话 snapshot | 既有会话保持已冻结 snapshot，但标注“新会话不可再使用” |

“声音身份正确”优先于“必须发出声音”。因此，用户明确选择的克隆音色发生故障时不得悄然变成默认声音。

## 4. 核心抽象

新增 `TtsRouter`，实现现有 TTS client 的最小接口：

```python
async def synthesize(self, text: str) -> bytes: ...
async def synthesize_snapshot(self, text: str, snapshot: TtsVoiceSnapshot) -> bytes: ...
def is_ready(self) -> bool: ...
def is_available(self) -> tuple[bool, str]: ...
def capabilities(self) -> dict[str, object]: ...
async def aclose(self) -> None: ...
```

`VoicePipeline` 只依赖这个接口，不负责判断引擎或读取模型配置。Router 的职责是：

- 根据 snapshot 的 backend family 解析目标 client；
- 对默认合成使用 `default_backend`；
- 维护按引擎划分的并发闸门和队列状态；
- 将底层错误转换为不泄漏路径、密钥或参考音频内容的稳定错误；
- 聚合每个引擎的 capability、availability、队列长度和最近错误摘要；
- 在应用关闭时依次关闭全部已实例化 client。

任何 client 仍负责自身模型线程亲和性。例如 `IndexTTS25MlxClient` 继续在自己的单 worker executor 内加载模型、构建 `SpeakerContext` 和合成；Router 不得把 MLX 对象跨线程传递。

## 5. 配置

建议新增显式路由配置，旧 `TTS_BACKEND` 在迁移期保留为默认 backend 的兼容别名：

```text
TTS_ROUTER_ENABLED=true
TTS_DEFAULT_BACKEND=piper
TTS_DEFAULT_FALLBACK_BACKEND=
TTS_ENABLED_BACKENDS=piper,indextts_mlx
TTS_USER_ENABLED_BACKENDS=piper
TTS_INDEXTTS_VALIDATED=false
TTS_ENGINE_QUEUE_SIZE=2
TTS_PREVIEW_QUEUE_SIZE=1
TTS_ENGINE_REQUEST_TIMEOUT_SECONDS=30
```

规则：

- `TTS_DEFAULT_BACKEND` 必须属于 `TTS_ENABLED_BACKENDS`；
- `TTS_USER_ENABLED_BACKENDS` 必须是已启用引擎的子集；它独立表达“模型/配置存在”与“已通过设备验收、可在用户界面选择”；
- `TTS_INDEXTTS_VALIDATED` 默认 `false`；即使 IndexTTS 已安装、可探测，也必须在目标设备完成 L2/L3 验收后才可在用户界面试听或选择。
- `TTS_ROUTER_ENABLED` 默认关闭，以保持仅配置旧 `TTS_BACKEND` 的既有部署行为；启用路由前必须显式配置默认与已启用引擎；
- fallback 只能是已启用、非默认的系统声音引擎，不能用于带 snapshot 的请求；
- `indextts_mlx` 必须在通过设备门禁后才可加入面向用户的 enabled 列表；
- 未启用的引擎不构造 client，也不出现在音色选择器；
- 后端配置、模型目录与凭据不经普通配置 API 回显。

## 6. 生命周期与资源治理

1. 应用启动时只注册 client factory，不加载大模型。
2. Router 首次收到该引擎请求时才创建 client；client 自己继续采用懒加载模型。
3. 每个引擎拥有独立的合成并发限制。IndexTTS 首期限制为 1；Piper 可以使用独立限制。
4. 预览请求优先级低于实时会话。实时会话队列满时，预览应返回 429 或被排队，不得抢占正在播报的会话。
5. 同一会话使用的 engine key 在 `start_session` 时冻结；Router 不在会话中重新读取全局默认 backend。
6. 应用关闭时拒绝新请求、取消等待队列、等待有限时间，然后调用每个 client 的 `aclose()`。

## 7. 健康与前端可见性

`/api/health` 的 TTS 区域应提供：

```json
{
  "default_backend": "piper",
  "fallback_backend": null,
  "engines": {
    "piper": {"available": true, "default": true, "clone": false},
    "indextts_mlx": {
      "available": false,
      "user_enabled": false,
      "clone": true,
      "reason": "设备验证尚未通过"
    }
  }
}
```

前端音色库和实时助手只能依据此结果显示可选引擎。音色元数据存在不代表对应引擎可用于会话。

## 8. 会话快照与审计

`Session` 除现有 `tts_voice_snapshot` 外，应冻结解析后的 `tts_engine_key`。实时会话事件、录音元数据和可选诊断记录至少包含：

```json
{
  "tts": {
    "engine": "indextts_mlx",
    "voice_id": "...",
    "voice_revision": 3,
    "reference_sha256": "...",
    "model_revision": "..."
  }
}
```

系统默认声音不含 voice 字段，但仍记录实际 engine 与 model revision。审计数据仅保存标识与摘要，不保存参考音频路径、完整授权备注或密钥。

## 9. 实施与验收

路由实现属于 `TTS-00A`，是实时音色选择 `TTS-10` 的硬依赖。

最小验收：

1. Piper 默认会话和 IndexTTS 指定音色会话可在同一应用进程内分别建立；
2. IndexTTS 模型未请求时不加载，Piper 会话不因其不可用而失败；
3. 指定 IndexTTS 音色失败时不会产生 Piper 音频；
4. 预览与实时会话使用独立的限流，实时会话优先；
5. 会话中新增 revision、修改默认 backend 或撤销授权不会替换已冻结的 engine/voice snapshot；
6. health 能逐引擎展示可用性，前端不显示无法路由的音色；
7. 关闭应用后所有已创建 client 的 executor 均被关闭。
