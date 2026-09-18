# Astra 真实设备验收清单

> 此清单补足 CI 与单元测试不能覆盖的设备、模型和网络边界。每项记录日期、设备、模型 ID、音频样本来源、结果和失败日志路径；不要把录音或密钥写入本文档。

## 前置条件

- [ ] Apple Silicon Mac mini 上的 Python 3.11 虚拟环境与模型路径可用。
- [ ] `ADMIN_TOKEN` 已配置，生产 HTTPS 场景已启用 `AUTH_COOKIE_SECURE=true`。
- [ ] `/api/health` 显示 ASR、TTS、VAD 与所选 LLM 的实际状态。
- [ ] 远端 LLM 的 `/v1/models` 与流式 `/v1/chat/completions` 均可从 Mac mini 访问。

## 实时助手

- [ ] 浏览器授予麦克风权限后，16 kHz PCM16 输入能产生 ASR、LLM 与 TTS 结果。
- [ ] 正常对话至少连续 10 轮，检查文本上下文、TTS 顺序和手动/语音打断。
- [ ] 在一次已完成生成后断网再恢复；重连后的首轮必须显示 ASR 文本、LLM token 和 TTS，不能被旧 generation 丢弃。
- [ ] 页面切到后台至少 60 秒再恢复，确认 AudioContext 恢复或给出明确的浏览器权限错误。
- [ ] 录制足够长的测试音频，确认浏览器内存不随完整录音线性增长，停止后可下载服务端封装的 WAV。

## 会议与训练闭环

- [ ] 提交短音频和长音频各一份，检查任务状态 WebSocket、`status.json`、`job.log` 与产物一致。
- [ ] 验证带重叠讲话的样本：分离/检测失败时仍保留混音 ASR 结果和失败原因。
- [ ] 在审核页播放、编辑并批准片段；训练数据列表只能展示含 approved 行的候选集。
- [ ] 启动、停止并重启训练任务，验证 SQLite 任务状态与实际子进程一致。

## 性能与质量记录

- [ ] 记录冷启动和热启动的首字、首音、整段会议耗时以及音频时长。
- [ ] 记录实时与会议任务期间的 RSS、Metal 内存、CPU、温度/降频和失败日志。
- [ ] 针对中文、多人、噪声和长音频分别人工复听，记录 ASR、分离、纪要和 TTS 的主观结论。
- [ ] 对每个失败注明是否可复现、输入时长、配置版本和模型 ID；不要以单次成功替代 P95 或长音频结论。

## VoiceStudio 路线图：模型真机验收（不计入代码完成度）

### P0：TTS 后端

- [ ] `TTS_BACKEND=mlx_audio`：普通话合成后人工播放 WAV，记录冷/热启动耗时、音频时长、RSS 与 Metal 内存。
- [ ] `TTS_BACKEND=cosyvoice`：仅在受控本地路径配置 `TTS_COSYVOICE_MODEL`、`TTS_COSYVOICE_PROMPT_WAV`、`TTS_COSYVOICE_PROMPT_TEXT`；确认 `/api/health` 的 `clone_configured=true`。
- [ ] 用同一参考音频分别执行 `TTS_COSYVOICE_LANGUAGE=zh` 与 `yue`，复听语言自然度、音色相似度和可懂度；记录模型版本和失败栈。
- [ ] 切回 `TTS_BACKEND=piper`，完成一轮实时对话回归，确认兜底没有加载 CosyVoice 依赖。

### P1/P2：ASR 与会议分离

- [ ] `ASR_BACKEND=funasr`：进行一轮实时对话，确认 ASR 文本、LLM 与 TTS 都完成，且没有启用会议说话人标签。
- [ ] 对三段已获授权的录音，以 MLX 路径和 FunASR 路径各跑一次会议 workflow；记录字错率、说话人一致性、端到端耗时和峰值内存。
- [ ] 含至少两位已注册说话人的录音中，核对 `Speaker N` 是否映射到注册声纹；映射不确定时必须保留原标签和置信原因。
- [ ] 切回 `ASR_BACKEND=mlx_audio` 重跑其中一段，确认原 VAD → ASR → 独立分离流程不变。
