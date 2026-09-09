# Astra

局域网语音助手 MVP：Mac mini 宿主机原生运行 FastAPI、MLX Whisper 和 Piper SDK，LLM 由 `192.168.3.18` 上的 `omlx` 或 `llama-server` 提供。Docker Compose 只运行前端 Nginx 反代。

## 本地后端测试

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
PYTHONPATH=backend .venv/bin/pytest -q backend/tests
```

## 文档

- [用户使用指引](docs/user-guide.md)：启动服务、上传音频、实时转写、流式修正和会议纪要。
- [开发指引](docs/development-guide.md)：项目结构、开发环境、测试、模型配置和扩展边界。
- [部署预检](deploy/README.md)：Mac mini、远端 LLM 和前端 Compose 验收。
- [Qwen3-ASR 微调接入设计](docs/qwen3-asr-finetuning-design.md)：会议数据闭环、训练参数、CUDA/Mac mini 可行性和模型发布门禁。
- [AGENT.md](AGENT.md)：代码代理和协作者在本仓库中的操作约定。

## Compose

```bash
cp .env.example .env
# 编辑 .env，填写远端模型名和本地语音模型路径
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
docker compose up -d
```

具体的远端 LLM 协议和 Mac mini 预检命令见 [`deploy/README.md`](deploy/README.md)。

## 会议 Workflow 工具

会议 Workflow 的模型下载、三段录音回归和本地 LLM 纪要工具统一放在
`scripts/meeting/`：

```bash
PYTHONPATH=backend .venv/bin/python scripts/meeting/download_and_test_workflow.py
PYTHONPATH=backend .venv/bin/python scripts/meeting/generate_llm_minutes.py
```

本地录音样本统一放在 `recordings/`，该目录只保留占位文件进 Git，实际音频
不提交。

纪要脚本默认读取 `/tmp/astra-workflow-test/report.json`，使用本机缓存的
`mlx-community/Qwen2.5-7B-Instruct-4bit`，不重复执行 ASR。
