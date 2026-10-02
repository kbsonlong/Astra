# TTS 目标 Mac 验收记录：2026-10-01

> 状态：模型安装、Piper/IndexTTS 冷热合成、连续 10 轮及试听 API 集成验证通过；使用真实获授权参考录音的浏览器复听、麦克风对话和重连验证仍待完成。正式配置中的 IndexTTS 门禁保持关闭。

## 设备基线

| 项目 | 结果 |
| --- | --- |
| 设备 | Apple M4（arm64） |
| 系统 | macOS 26.6.2（25G83） |
| 内存 | 16 GiB |
| Python | 3.11.6 |
| 当前提交 | `b1c8e80` |
| 工作目录可用空间 | 安装后约 13 GiB（228 GiB 中约 189 GiB 已用） |

## 运行时预检

| 项目 | 结果 |
| --- | --- |
| 后端 `http://127.0.0.1:8000/api/health` | 集成测试临时启动通过，测试结束后已关闭 |
| `TTS_BACKEND` | `piper` |
| `TTS_ROUTER_ENABLED` | `true` |
| `TTS_INDEXTTS_VALIDATED` | `false`（符合安全默认值） |
| Piper 模型文件 | `models/zh_CN-huayan-medium.onnx`，约 60 MB |
| IndexTTS MLX 模型目录 | HF 缓存快照 `660a09b98cf667130831e73d7d49d66de1236645` |
| ASR / VAD 模型路径 | 存在 |

## 已完成验证

- Piper 合成：冷调用 0.482 秒、热调用 0.087 秒；WAV 为 22.05 kHz 单声道。
- IndexTTS 缓存快照推理：合规 7.036 秒参考音频；冷调用 15.252 秒、热调用 6.802 秒；输出均为 22.05 kHz 单声道。
- IndexTTS 连续 10 轮：全部成功，输出为 22.05 kHz 单声道；首轮 13.321 秒，热调用中位数 8.199 秒、范围 7.821–8.293 秒；峰值进程内存 5,359,239,168 字节（约 4.99 GiB）。使用 7.906 秒 Piper 合成参考音频，模型固定 revision 与配置一致。
- IndexTTS 试听 API：临时测试库完成创建（201）、确认授权（200）、启用（200）；正式门禁关闭时预览返回 503。只在临时测试进程打开门禁后，真实预览返回 200，耗时 13.681 秒，并带回 `voice_id`、revision 和模型 revision。
- 健康接口报告 Piper 可用；IndexTTS 技术可用但 `validated=false`、`user_enabled=false`、`user_reason=device_validation_required`。
- 修正了两个真机暴露问题：Piper SDK 需要 `wave.Wave_write` 对象；IndexTTS 健康探测不能在 ASGI 线程导入 MLX SDK，否则会与推理线程重复注册原生枚举。
- 沙箱内没有 Metal 设备；模型推理和试听 API 在获准访问本机 Metal 的执行环境内完成。
- 代码回归：后端全量测试 212 passed（Metal 可访问执行）；前端 11 passed，生产构建通过；TTS 相关后端专项 51 passed。

## 尚未完成与下一步

1. 在浏览器中用真实且已授权的 5–15 秒参考录音试听，并由人复听音质。本记录中的参考录音由 Piper 自动合成，只用于技术验证。
2. 完成实时麦克风对话、语音打断、断网重连、授权撤销与 Piper 回退测试；记录首音/尾音 P50/P95、RSS、Metal 内存及失败日志。
3. 正式多引擎配置保持以下门禁：

   ```dotenv
   TTS_ROUTER_ENABLED=true
   TTS_ENABLED_BACKENDS=piper,indextts_mlx
   TTS_USER_ENABLED_BACKENDS=piper,indextts_mlx
   TTS_DEFAULT_BACKEND=piper
   TTS_INDEXTTS_VALIDATED=false
   ```

4. 只有以上 L2/L3 指标、真实参考音频浏览器试听和人工复听全部完成后，管理员才可将 `TTS_INDEXTTS_VALIDATED` 设为 `true`。
