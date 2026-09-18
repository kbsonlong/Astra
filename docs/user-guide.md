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
- 生成会议纪要：调用 `/api/meeting/process`，提交后立即得到 `task_id`，页面通过 `WS /api/meeting/{task_id}/events` 接收处理状态；
- 会议主题：可选填写，用于辅助会议摘要。

实时助手通过 `/ws` 使用 16 kHz PCM16 传输音频。全部麦克风帧会同时写入 Mac mini 的服务端录音目录；只有聆听窗口内的帧进入 ASR，因此浏览器不会缓存整段音频，也不会把播报期间的帧误送入下一轮转写。点击“停止通话”后，服务端将录音封装为 WAV；“下载录音”按钮会把该文件下载到浏览器所在电脑。单次服务端录音默认最多 2 GiB，默认保留 7 天；管理员可通过 `.env` 中的 `REALTIME_RECORDING_*` 配置调整。

支持 `.m4a`、`.wav`、`.mp3`、`.flac`、`.aac`、`.mov` 和 `.mp4`。普通会议任务最大 500 MB，声纹注册样本最大 20 MB。

会议任务完成后，报告默认写入 `~/Astra/meetings/<task_id>/`，包括 `report.md`、`transcript.txt`、`meta.json`、`status.json` 和 `job.log`。录音样本统一放在项目根目录的 `recordings/`，该目录中的真实音频不会提交到 Git。

若配置了 `ADMIN_TOKEN`，先在网页登录；令牌只在登录请求中提交，后续请求使用同源 HttpOnly 会话 Cookie。没有配置令牌的兼容模式只适用于可信局域网。

## 常见问题

### 健康检查失败

先确认后端进程仍在运行，再检查模型路径和 `.env` 配置。`/api/health` 的响应会区分配置缺失、模型未加载和服务正常。

### 流式修正不可用

确认 `LLM_CORRECTION_ENABLED=true`、`LLM_MODEL` 已配置，并且远端 LLM 提供 OpenAI 兼容的 `/v1/chat/completions` 接口。会议纪要默认关闭 LLM 清洗，只应用已确认的确定性规则。

### 会议任务长时间 processing

查看任务目录中的 `job.log`。会议处理运行在独立进程中，任务状态以 `status.json` 为准；不要重复提交同一大文件来替代排查日志。
