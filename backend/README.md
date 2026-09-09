# Backend host setup

FastAPI、MLX Audio 和 Piper SDK 必须在 Mac mini 宿主机原生运行。MLX 需要 Apple Silicon 运行环境，因此 backend 不提供 Linux Docker 镜像。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`ASR_MODEL` 使用 `mlx-audio` 模型标识或本地模型目录，默认值为 `mlx-community/Qwen3-ASR-0.6B-4bit`。`TTS_MODEL_PATH` 指向 Piper `.onnx` voice 文件。ASR 模型首次加载发生在第一次请求时，之后复用已加载模型；部署前应先用真实音频做预热和耗时测量。

超过 `ASR_LONG_AUDIO_THRESHOLD_SECONDS` 的音频会按 `ASR_CHUNK_DURATION_SECONDS` 切片，每个切片独立转写后拼接。`ASR_MAX_TOKENS` 是每个切片的输出预算，长会议建议设置为 512 或 1024。

## 会议 Workflow

`POST /api/meeting/process` 使用独立 worker 执行完整链路：

```text
Silero VAD -> ASR -> punctuation -> speaker diarization -> LLM summary
```

可通过 `GET /api/meeting/prompt-templates` 查看纪要模板，并在上传时以
`prompt_template` 表单字段选择内置或自定义模板。自定义模板可通过同一资源的
`POST`、`PUT /{template_id}`、`DELETE /{template_id}` 接口管理；内置模板
`standard`、`decisions`、`concise` 只读。模板只影响 LLM 纪要阶段，不影响 ASR、
标点、声纹分离或确定性纠错；任务的选择会记录在 `meta.json` 中。

自定义模板默认保存到 `~/.astra/meeting-prompt-templates.json`，可通过
`MEETING_PROMPT_TEMPLATES_PATH` 修改路径。每个模板包含名称、说明、分段提取系统
提示词和最终合并系统提示词。

四阶段编排位于 `app/core/workflow.py`，每个输出段保留 VAD 的 `start/end` 时间戳。默认 `PUNCTUATION_ENGINE=passthrough` 保持最小依赖；安装 FunASR 后设置 `PUNCTUATION_ENGINE=funasr`，并通过 `PUNCTUATION_MODEL`、`PUNCTUATION_DEVICE` 选择标点模型和设备。当前 SD 实现为 `resemblyzer` + Ward 聚类，可通过相同的 `SpeakerDiarizationStage` 契约替换为 CAM++ 等模型。

声纹档案保存在 `SPEAKER_STORE_PATH` 指定的 SQLite 中。`SPEAKER_MAX_SPEAKERS` 控制单场会议的最大聚类人数，默认 32，可按团队规模调整；未匹配到正式档案的声纹会进入 `pending_review`，管理员在前端站内提醒中改名并点击“审核通过”后，才会参与后续会议匹配。

## 后台训练

训练参数保存后，可在前端点击“启动训练”。后端通过 `mlx-tune` SDK 在独立进程中执行，
不启动 Python 命令行；日志和适配器写入 `output_dir`。项目 `.venv` 需要安装 `mlx-tune`：

```bash
.venv/bin/pip install 'mlx-tune @ git+ssh://git@github.com/kbsonlong/mlx-tune.git@d5bcb880034078a78f0db51684e5783db44fbfa6'
```

训练 JSONL 每行需要 `audio` 和 `text` 字段，音频路径支持 `~`。任务状态可通过
`GET /api/training/status` 查询，停止使用 `POST /api/training/stop`。
