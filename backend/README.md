# Backend host setup

FastAPI、MLX Audio 和 Piper SDK 必须在 Mac mini 宿主机原生运行。MLX 需要 Apple Silicon 运行环境，因此 backend 不提供 Linux Docker 镜像。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`ASR_MODEL` 使用 `mlx-audio` 模型标识或本地模型目录，默认值为 `mlx-community/Qwen3-ASR-0.6B-4bit`。`TTS_MODEL_PATH` 指向 Piper `.onnx` voice 文件。ASR 模型首次加载发生在第一次请求时，之后复用已加载模型；部署前应先用真实音频做预热和耗时测量。
