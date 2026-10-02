# TTS 音色库发布与验收说明

> 发布范围：TTS-00A 至 TTS-11。本文记录代码级验证结果与仍需在目标 Mac 完成的设备验收，二者不可互相替代。

## 发布前配置

音色库依赖多引擎路由；Piper 保持系统默认回退，IndexTTS 必须经过显式门禁才会对用户开放：

```dotenv
TTS_ROUTER_ENABLED=true
TTS_ENABLED_BACKENDS=piper,indextts_mlx
TTS_USER_ENABLED_BACKENDS=piper,indextts_mlx
TTS_DEFAULT_BACKEND=piper
TTS_INDEXTTS_VALIDATED=false
```

- 初次发布保持 `TTS_INDEXTTS_VALIDATED=false`。此时可以创建、授权、启用 IndexTTS 音色，但不能试听或在实时助手中选择。
- 只有在目标 Mac 完成下面的设备验收，并由管理员明确将该值改为 `true` 后，IndexTTS 才会进入用户可选项。
- `/api/health` 的 `tts.engines.indextts_mlx` 同时提供 `available`、`user_enabled`、`validated` 与原因；不能仅以模型目录存在判断可发布。

## 管理员发布步骤

1. 备份现有 `TTS_VOICE_STORE_PATH` 数据库与 `TTS_VOICE_AUDIO_DIR` 目录；不要迁移或混入会议声纹档案。
2. 配置并启动后端后，检查 `GET /api/health`：Piper 必须可用；IndexTTS 未验证时应显示 `user_reason=device_validation_required`。
3. 打开“音色库”，用已获授权的 5–15 秒单声道 PCM16 WAV 创建测试音色；完成授权确认。
4. 在 IndexTTS 未验证阶段，确认“生成试听”和实时助手选择项均不可用，而系统默认 TTS 仍可开始会话。
5. 真机验证通过后，设置 `TTS_INDEXTTS_VALIDATED=true` 并重启服务；重新检查健康接口，再对同一音色执行试听和实时会话验收。
6. 若撤销授权，确认音色立即不再出现在新会话选择器中；历史会话仅保留版本标识，不恢复播放或合成权限。

## 已执行的代码级验证（2026-10-01）

| 检查 | 结果 |
| --- | --- |
| `PYTHONPATH=backend .venv/bin/python -m pytest -q backend/tests` | 212 passed with Metal access |
| `cd frontend && npm test` | 4 files, 11 passed |
| `cd frontend && npm run build` | passed |
| `git diff --check` | passed |

`.venv/bin/pytest` 的 shebang 可能仍指向旧工作目录；验收脚本应使用 `.venv/bin/python -m pytest`，避免解释器路径漂移。

## 目标 Mac 真机验收（发布门禁）

已完成 Piper 冷/热合成、IndexTTS 缓存快照冷/热合成，以及音色创建、授权、启用、门禁拒绝和真实试听 API 链路。设备和性能数据见 [2026-10-01 设备验收记录](tts-device-validation-2026-10-01.md)。

以下项目仍未完成，完成前不能开启 IndexTTS 用户门禁：

- [x] 记录设备型号、macOS、Python、IndexTTS 模型仓库与固定 revision。
- [x] 完成 IndexTTS 冷/热短句及连续 10 轮推理；结果与峰值内存见设备记录。
- [ ] 完成语音打断、首音/尾音 P50/P95 和 Metal 长时资源记录。
- [ ] 在实时助手中选择已授权 rN，确认 WebSocket `start_session` 固定传递该 `voice_id + voice_revision`；新建 rN+1 不改变进行中会话。
- [ ] 断网重连后确认会话继续使用同一版本；撤销授权后确认页面告警且新会话无法选择该音色。
- [ ] 切回系统默认 Piper，确认 IndexTTS 不可用或被关闭时实时对话仍可用。

详细跨设备项目见 [真实设备验收清单](acceptance-checklist.md)。

当前目标 Mac 的实际预检结果见 [2026-10-01 设备验收记录](tts-device-validation-2026-10-01.md)。
