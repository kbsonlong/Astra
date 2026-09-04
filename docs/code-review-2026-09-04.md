# Python 代码变更评审报告

> 评审时间：2026-09-04
> 最新提交：`ba70f12 feat: add streaming ASR correction`
> 未提交变更：新增 SherpaSenseVoice ASR 引擎 + 多引擎切换
> 测试结果：19/19 通过

---

## 一、变更范围

本次评审覆盖两个层次的变更：

| 变更批次 | 状态 | 涉及文件 | 核心内容 |
|----------|------|----------|----------|
| 已提交 `ba70f12` | committed | 5 个 Python 文件 | SSE 流式 ASR 纠错端点 + LLM 客户端扩展 |
| 未提交 (working tree) | unstaged | 4 个 Python 文件 + requirements | SherpaSenseVoice ASR 引擎 + 多引擎切换 |

---

## 二、逐模块评审

### 2.1 `http_routes.py` — 新增 SSE 流式纠错端点

**变更内容**：新增 `POST /api/transcribe/stream` 端点，先 ASR 转写，再用 LLM 流式纠错，通过 SSE 推送事件。

**正面评价**：
- SSE 辅助函数 `_sse()` 简洁实用，`ensure_ascii=False` 保证中文正确输出
- 降级逻辑设计良好：LLM 纠错失败时回退到原始 ASR 文本（`fallback: True`），不阻断用户体验
- `StreamingResponse` 的 headers 配置得当（`X-Accel-Buffering: no` 防止 Nginx 缓冲）
- 前置校验完整：ASR 缺失、LLM 缺失、功能未启用、模型未配置都有明确的 503 错误信息

**问题与建议**：

1. **[P1] `stream_chat` 的 `max_tokens` / `chat_template_kwargs` 参数直接透传，但 OpenAI 官方 API 不支持 `chat_template_kwargs`**
   - `chat_template_kwargs` 是 llama.cpp / llama-server 的扩展参数，发给标准 OpenAI API 会被忽略或报错
   - 当前代码注释说明了使用 `omlx` 或 `llama-server`，所以实际可行
   - **建议**：在 `llm_client.py` 的 `stream_chat` docstring 中注明 `chat_template_kwargs` 仅兼容 llama-server 系列，避免未来切换到其他 LLM 服务时的困惑

2. **[P2] ASR 转写失败时 HTTPException 在 `events()` 生成器之外抛出，返回 503 — 这是正确的；但如果 ASR 成功后 LLM 流式纠错中途客户端断开，`events()` 生成器会被 `GeneratorExit` 中止，`corrected` 中的部分结果会丢失**
   - 这是 SSE 流式响应的固有特性，MVP 阶段可接受
   - **建议**：在 `events()` 中捕获 `GeneratorExit`，记录已消费但未推送的 token 量用于监控

3. **[P3] `_sse` 函数名使用了下划线前缀暗示私有，但定义在模块级别**
   - 不影响功能，但 `sse_event` 或 `format_sse` 更符合 Python 模块级函数命名惯例

4. **[P3] `events()` 生成器内的 `corrected` 变量在 except 分支后不再使用，但仍赋值**
   - except 分支中 `corrected` 未被清空，但由于 `else` 分支才使用它，逻辑正确
   - 纯粹是代码可读性的小瑕疵

### 2.2 `config.py` — 新增纠错配置 + Sherpa 引擎配置

**变更内容**：新增 `_bool_env` 辅助函数，新增 6 个 LLM 纠错配置项和 5 个 Sherpa 配置项。

**正面评价**：
- `_bool_env` 实现干净，支持 `1/true/yes/on` 四种真值
- 所有新增配置项都有合理的默认值，开箱即用
- `llm_correction_system_prompt` 内置了实用的术语映射规则（后视→host 等），体现了对业务场景的理解
- frozen dataclass + `from_env` 的模式保持一致

**问题与建议**：

1. **[P1] `sherpa_use_itn` 和 `sherpa_auto_language` 使用 `_bool_env` 读取，但 `.env.example` 中写的是 `true`/`false` — `_bool_env` 只识别真值集合，`false` 会正确返回 `False`，但如果用户写 `False`（大写）也 OK（因为 `.lower()`）。然而如果用户写 `0` 或 `no`，这些不在真值集合中，会返回 `False` — 行为正确但不够显式。**
   - **建议**：在 `_bool_env` 的 docstring 或注释中说明 falsy 值的范围

2. **[P3] `.gitignore` 变更删除了末尾换行符（`\ No newline at end of file`）**
   - 小问题但会影响 diff 的整洁度

3. **[P3] `asr_engine` 默认值为 `"mlx"`，但 `_build_asr_client` 中用 `(current.asr_engine or "mlx").lower()` — `or` 在空字符串时也触发，设计上是双保险，可接受**

### 2.3 `llm_client.py` — 扩展 `stream_chat` 参数

**变更内容**：新增 `max_tokens` 和 `chat_template_kwargs` 两个可选参数。

**正面评价**：
- 条件添加到 payload 的方式正确：`if max_tokens is not None` 和 `if chat_template_kwargs is not None`，不影响原有调用方
- `chat_template_kwargs` 做了 `dict()` 浅拷贝，避免外部修改影响 payload
- 类型标注准确：`int | None` 和 `Mapping[str, Any] | None`

**问题与建议**：

1. **[P2] `max_tokens` 参数名与 OpenAI API 的 `max_tokens` 一致，但 OpenAI 已开始用 `max_completion_tokens` 替代。llama-server 兼容 `max_tokens`，所以当前可用。**
   - **建议**：无即时风险，但在注释中注明兼容性

2. **[P3] `dict(chat_template_kwargs)` 是浅拷贝，嵌套对象仍可被外部修改 — 但 `chat_template_kwargs` 通常是扁平的 `{"enable_thinking": False}`，实际无影响**

### 2.4 `main.py` — ASR 引擎工厂 + 配置端点扩展

**变更内容**：新增 `_build_asr_client()` 工厂函数，支持 mlx 和 sherpa 两种引擎切换；`/api/config` 端点新增 sherpa 相关配置输出。

**正面评价**：
- 工厂函数提取得当，`create_app` 中的 pipeline 构造更简洁
- 引擎匹配用别名集合 `{"sherpa", "sherpa-sensevoice", "sensevoice", "sense-voice"}`，用户体验友好
- `/api/config` 端点完整暴露了所有 sherpa 配置，便于运维排查
- `_build_asr_client` 返回类型标注为 `object`，虽然不够精确但考虑到两个客户端无共同基类，是合理的折中

**问题与建议**：

1. **[P1] `OpenAICompatLLMClient.aclose()` 仍未被调用 — `main.py` 中未注册 FastAPI lifespan 事件来关闭 LLM 客户端的 httpx 连接池。这不是本次变更引入的问题，但本次变更新增了对 LLM 客户端的使用（纠错端点），使该问题更突出。**
   - **建议**：
     ```python
     @asynccontextmanager
     async def lifespan(app: FastAPI):
         yield
         if hasattr(app.state, "pipeline") and app.state.pipeline:
             llm = getattr(app.state.pipeline, "llm", None)
             if llm and hasattr(llm, "aclose"):
                 await llm.aclose()
     app = FastAPI(title="Astra API", version="0.1.0", lifespan=lifespan)
     ```

2. **[P3] `_build_asr_client` 返回 `object` 类型 — 后续可考虑定义 `ASRClient` Protocol（`transcribe` + `is_ready`）替代，获得更好的类型安全**

### 2.5 `asr_client.py` — 新增 `SherpaSenseVoiceAsrClient`（未提交）

**变更内容**：新增 233 行的 `SherpaSenseVoiceAsrClient` 类，实现 sherpa-onnx SenseVoice 引擎的 ASR 客户端。

**正面评价**：
- 与 `MlxAudioAsrClient` 接口一致（`is_ready()` + `transcribe()`），可无缝替换
- 依赖注入设计延续一致：`_create_recognizer`、`_read_wav`、`_resample` 三个内部函数均可通过构造函数参数注入，测试时不需安装 sherpa-onnx
- `_ensure_model_files` 使用 glob 模式匹配模型文件，容错性好
- WAV 读取 `_default_read_wav` 支持多种采样位宽（8/16/32bit）和多声道降混
- 重采样 `_default_resample` 用线性插值实现，虽简单但功能正确
- 长音频分块逻辑与 MLX 客户端一致
- 异常处理统一包装为 `ASRClientError`
- `_decode_blocking` 中对 `result` 的多种类型（dict、对象属性、JSON 字符串）都做了适配，容错性强

**问题与建议**：

1. **[P1] `_decode_blocking` 中 `import sherpa_onnx` 在每次调用时执行 — 虽然 Python 会缓存模块导入，但在 `asyncio.to_thread` 中高频调用仍有不必要的导入查找开销。**
   - **建议**：在类级别或 `_get_or_create_recognizer` 中一次性导入并缓存

2. **[P1] `numpy` 在 `_default_read_wav` 和 `_default_resample` 中使用，但 `requirements.txt` 中新增的 `numpy>=1.24,<2` 是正确的。不过 `_default_read_wav` 中的 `import numpy as np` 在函数内部执行 — 每次调用都重新导入。**
   - **建议**：将 `import numpy as np` 提到模块顶部，或放到 `__init__` 中一次性导入并缓存为 `self._np`

3. **[P2] `_default_read_wav` 中 `np.clip(pcm, -1.0, 1.0, out=pcm)` — `out` 参数要求 `pcm` 是 ndarray 且 dtype 匹配。此时 `pcm` 已通过 `pcm.astype(np.float32)` 转换，应该没问题，但 `np.iinfo(pcm.dtype).max + 1.0` 对于 `int16` 是 `32768.0`，除法后范围是 `[-1.0, 1.0]`，clip 是防御性的，合理。**

4. **[P2] `language` 字段处理有微妙之处：构造函数中 `self.language = "" if auto_language else language`，但 `_default_create_recognizer` 中 `if language: setattr(recognizer, "_sense_voice_language", language)` — 用 `setattr` 设置私有属性来传递语言参数，这是 hack 做法。**
   - 原因：`sherpa_onnx.OfflineRecognizer.from_sense_voice` 的构造参数中没有 `language` 参数
   - **建议**：添加注释说明为何用 setattr，避免未来维护者困惑

5. **[P2] `is_ready()` 会调用 `_ensure_model_files()` 来检查文件是否存在，但这会设置 `self._model_path` 和 `self._tokens_path` 作为副作用 — 在"只读检查"中产生写副作用。**
   - **建议**：将文件检查逻辑与路径赋值分离，或在 `is_ready` 中用纯只读方式检查

6. **[P2] 缺少测试 — `SherpaSenseVoiceAsrClient` 没有对应的测试文件。它有完善的依赖注入设计（`_create_recognizer`、`_read_wav`、`_resample`），完全可以像 `test_asr_client.py` 那样编写注入式测试。**
   - **建议**：新增 `test_sherpa_asr_client.py`，至少覆盖：正常转写路径、空音频、长音频分块、模型文件缺失、SDK 导入失败

7. **[P3] `import glob` 和 `from pathlib import Path` 新增 — `glob.glob` 用于模式匹配，`Path.resolve` 用于路径规范化，两个模块混用可接受但不统一。**
   - **建议**：统一用 `Path.glob()` 替代 `glob.glob()`

8. **[P3] `_default_resample` 是线性插值重采样 — 音质不如多相滤波器，但对于 ASR 输入通常足够。可在注释中说明这是简化实现。**

### 2.6 `health.py` — ASR 模式探测

**变更内容**：根据 ASR 客户端类名返回不同的 `asr.mode` 和新增 `asr.engine` 字段。

**问题与建议**：

1. **[P1] `type(asr_client).__name__ == "SherpaSenseVoiceAsrClient"` — 用类名字符串匹配做类型判断是脆弱的。如果重命名类或继承子类，匹配会失效。**
   - **建议**：用 `isinstance` 检查，或更好——在两个 ASR 客户端上定义一个共同的 `engine_name` 属性：
     ```python
     class MlxAudioAsrClient:
         engine_name = "mlx-sdk"
     class SherpaSenseVoiceAsrClient:
         engine_name = "sherpa-sensevoice-onnx"
     ```
     然后 `asr_mode = getattr(asr_client, "engine_name", "unknown")`

### 2.7 `requirements.txt` — 新增依赖

**变更内容**：新增 `numpy>=1.24,<2` 和 `sherpa-onnx>=1.12.37`。

**正面评价**：
- `numpy` 的版本上限 `<2` 是明智的，numpy 2.x 有 breaking changes
- `sherpa-onnx>=1.12.37` 使用 `>=` 而非 `==`，因为 ASR 引擎是可选的

**问题与建议**：

1. **[P2] `sherpa-onnx` 应该是可选依赖 — 当用户选择 MLX 引擎时不需要安装 sherpa-onnx。当前放在 `requirements.txt` 中会导致必须安装。**
   - **建议**：将 `sherpa-onnx` 和 `numpy` 移到 `requirements-sherpa.txt` 或用 `extras_require`：
     ```txt
     # requirements.txt (核心依赖)
     fastapi==0.116.1
     httpx==0.28.1
     ...

     # requirements-sherpa.txt (可选)
     -r requirements.txt
     numpy>=1.24,<2
     sherpa-onnx>=1.12.37
     ```

---

## 三、测试评审

### 3.1 已有测试

| 测试文件 | 新增/变更 | 评价 |
|----------|-----------|------|
| `test_transcribe_route.py` | +1 测试 + FakeLLM | 覆盖了 SSE 端点的成功路径，验证了事件类型和内容 |

**正面评价**：
- `FakeLLM` 的 `stream_chat` 断言了 `max_tokens`、`chat_template_kwargs` 和传入的 messages 内容，验证了参数透传
- SSE 响应的断言覆盖了 `asr_final`、`correction_token`、`correction_final`、`[DONE]` 四种事件

**缺失场景**：
- **[P2] 缺少 LLM 纠错失败的降级测试** — 应注入一个抛出 `LLMClientError` 的 FakeLLM，验证 `fallback: True` 事件
- **[P2] 缺少 LLM 功能禁用的测试** — `llm_correction_enabled=False` 时应返回 503
- **[P2] 缺少 SherpaSenseVoiceAsrClient 的测试文件**
- **[P3] 缺少 health 端点返回 sherpa mode 的测试**

### 3.2 测试运行结果

```
19 passed, 2 warnings in 0.59s
```

2 个 warning 来自 starlette/httpx 的弃用提示，不影响功能。

---

## 四、安全评审

| 风险 | 严重度 | 说明 |
|------|--------|------|
| `chat_template_kwargs` 暴露内部模型参数 | 低 | 仅在 llama-server 环境下有意义，不构成安全风险 |
| SSE 端点无文件大小限制 | 中 | 与 `/transcribe` 相同的既有问题，非本次引入 |
| `/api/config` 新增暴露 sherpa 配置 | 低 | 路径和线程数等不敏感信息 |
| sherpa 模型路径可被外部探测 | 低 | `/api/config` 暴露了 `sherpa_model_dir`，但不包含凭据 |

---

## 五、综合评价

### 优点

1. **SSE 流式纠错设计优雅** — 先 ASR 后 LLM 纠错的双阶段设计，降级到原始文本的 fallback 逻辑完善
2. **多 ASR 引擎架构扩展干净** — 工厂模式 + 接口一致的客户端设计，新增引擎不改调用方
3. **依赖注入延续一致** — SherpaSenseVoiceAsrClient 的三个注入点（recognizer/read_wav/resample）设计到位
4. **配置项设计合理** — 所有新增配置都有默认值和 `.env.example` 文档
5. **类型标注全面** — 新增代码的 type hints 完整

### 需要改进

| 优先级 | 问题 | 涉及文件 |
|--------|------|----------|
| P1 | LLM 客户端 `aclose()` 仍未在 lifespan 中调用 | `main.py` |
| P1 | `health.py` 用类名字符串匹配做类型判断 | `health.py` |
| P1 | `SherpaSenseVoiceAsrClient` 缺少测试 | `tests/` |
| P1 | `sherpa-onnx` 应为可选依赖而非强制安装 | `requirements.txt` |
| P2 | 缺少 LLM 纠错失败降级路径的测试 | `test_transcribe_route.py` |
| P2 | `setattr` 传 language 的 hack 缺少注释 | `asr_client.py` |
| P2 | `is_ready()` 有写副作用 | `asr_client.py` |
| P3 | `_sse` 命名、numpy 模块级导入、glob/Path 混用 | 多文件 |

### 结论

本次变更是两个功能特性的叠加：**SSE 流式 ASR 纠错** + **SherpaSenseVoice 多引擎支持**。两者都是合理的功能演进，代码质量在 MVP 级别项目中属上乘。主要改进方向集中在测试覆盖补全（Sherpa 客户端 + 降级路径）、类型判断方式（用属性/Protocol 替代类名字符串匹配）、以及可选依赖管理上。
