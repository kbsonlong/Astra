# Astra

局域网语音助手 MVP：Mac mini 本地运行 Web、FastAPI、whisper.cpp 和 Piper，LLM 由 `192.168.3.18` 上的 `omlx` 或 `llama-server` 提供。

## 本地后端测试

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
PYTHONPATH=backend .venv/bin/pytest -q backend/tests
```

## Compose

```bash
cp .env.example .env
# 编辑 .env，填写远端模型名和已验证的固定镜像
docker compose up -d
```

具体的远端 LLM 协议和 Mac mini 预检命令见 [`deploy/README.md`](deploy/README.md)。
