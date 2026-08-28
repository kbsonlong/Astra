# Astra 部署预检

Mac mini 宿主机原生运行 FastAPI、MLX Whisper 和 Piper SDK，Compose 只运行前端 Nginx。LLM 必须在 `192.168.3.18` host 上运行，并向局域网暴露 OpenAI 兼容接口。

## 环境变量

复制一份环境文件并填写远端模型名、本地 ASR 模型和 Piper voice 路径：

```bash
cp .env.example .env
docker compose config
docker compose up -d
```

后端依赖 `mlx-whisper==0.4.3` 和 `piper-tts==1.7.0`，启动前必须完成 Python 依赖安装，并确保 `TTS_MODEL_PATH` 指向本地 Piper `.onnx` voice 文件。

## 远端 LLM 验收

`omlx` 或 `llama-server` 的启动参数由远端机器维护，但最终必须满足同一协议：

```bash
curl -fsS http://192.168.3.18:8000/v1/models
curl -fsS http://192.168.3.18:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"ping"}],"stream":true}'
```

如果选定的 `omlx` 启动方式没有直接提供该接口，应在 `192.168.3.18` 增加适配层；Mac mini FastAPI 不接入运行时私有协议。

## Mac mini 验收

```bash
docker compose ps
curl -fsS http://127.0.0.1:8080/
curl -fsS http://127.0.0.1:8080/api/health
curl -fsS http://127.0.0.1:8000/api/health
```
