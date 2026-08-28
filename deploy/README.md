# Astra 部署预检

Mac mini 的 Compose 只运行前端、FastAPI、whisper.cpp 和 Piper。LLM 必须在 `192.168.3.18` host 上运行，并向局域网暴露 OpenAI 兼容接口。

## 环境变量

复制一份环境文件并填写已经验证过的固定镜像和远端模型名：

```bash
cp .env.example .env
docker compose config
docker compose up -d
```

`WHISPER_IMAGE`、`PIPER_IMAGE` 必须是已经核验的固定 tag 或 digest，不能使用 `latest`。镜像的 `/health` 和接口契约必须与设计文档一致。

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
docker stats --no-stream
```
