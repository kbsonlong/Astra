# Astra 用户使用指引

## 适用环境

Astra 是局域网语音助手 MVP。后端依赖 Apple Silicon Mac mini 原生运行，前端通过 Docker Compose 提供访问入口。使用前需要准备：

- macOS + Apple Silicon；
- Python 3.11；
- 本地 ASR、VAD 和 Piper TTS 模型；
- 可从 Mac mini 访问的 OpenAI 兼容 LLM 服务。

## 启动服务

首次使用时，在项目根目录执行：

```bash
cp .env.example .env
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
```

编辑 `.env`，至少确认 `LLM_BASE_URL`、`LLM_MODEL`、`ASR_MODEL`、`VAD_MODEL` 和 `TTS_MODEL_PATH`。然后分别启动后端和前端：

```bash
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
docker compose up -d
```

浏览器打开 `http://<Mac-mini-IP>:8080/`。后端首次加载模型可能较慢，建议先检查：

```bash
curl -fsS http://127.0.0.1:8000/api/health
curl -fsS http://127.0.0.1:8080/api/health
```

## 使用网页功能

在页面选择音频文件后，可以使用以下功能：

- 普通转写：调用 `/api/transcribe`，等待 JSON 结果；
- 流式修正：调用 `/api/transcribe/stream`，先返回 ASR 文本，再以 SSE 推送纠错结果；
- 生成会议纪要：调用 `/api/meeting/process`，提交后立即得到 `task_id`，再轮询 `/api/meeting/{task_id}`；
- 会议主题：可选填写，用于辅助会议摘要。

支持 `.m4a`、`.wav`、`.mp3`、`.flac`、`.aac`、`.mov` 和 `.mp4`。普通会议任务最大 500 MB，声纹注册样本最大 20 MB。

会议任务完成后，报告默认写入 `~/Astra/meetings/<task_id>/`，包括 `report.md`、`transcript.txt`、`meta.json`、`status.json` 和 `job.log`。录音样本统一放在项目根目录的 `recordings/`，该目录中的真实音频不会提交到 Git。

## 常见问题

### 健康检查失败

先确认后端进程仍在运行，再检查模型路径和 `.env` 配置。`/api/health` 的响应会区分配置缺失、模型未加载和服务正常。

### 流式修正不可用

确认 `LLM_CORRECTION_ENABLED=true`、`LLM_MODEL` 已配置，并且远端 LLM 提供 OpenAI 兼容的 `/v1/chat/completions` 接口。会议纪要默认关闭 LLM 清洗，只应用已确认的确定性规则。

### 会议任务长时间 processing

查看任务目录中的 `job.log`。会议处理运行在独立进程中，任务状态以 `status.json` 为准；不要重复提交同一大文件来替代排查日志。
