# Astra 部署预检

Mac mini 宿主机原生运行 FastAPI、MLX Whisper 和 Piper SDK，Compose 只运行前端 Nginx。LLM 必须在 `192.168.3.18` host 上运行，并向局域网暴露 OpenAI 兼容接口。

## 环境变量

复制一份环境文件并填写远端模型名、本地 ASR 模型、Piper voice 路径和管理员认证凭据：

```bash
cp .env.example .env
# 生成两个不同的高熵值，分别填入 ADMIN_TOKEN 与 ADMIN_SESSION_SECRET。
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
# Mac Docker Desktop 下默认 host.docker.internal:8000 指向宿主机原生 FastAPI。
# 若后端不在本机，设置为目标地址（不含 http://），例如 API_UPSTREAM=192.168.3.18:8001。
# NGINX_CLIENT_MAX_BODY_SIZE 必须不小于预期会议录音，且不超过后端允许的 500MB。
docker compose config --quiet
docker compose up -d
```

`ASTRA_CONFIG_PATH` 指定运行时配置文件路径，默认是当前目录的 `.env`；若服务由 systemd、launchd 或其他工作目录启动器运行，建议填写绝对路径。管理设置保存 LLM 配置时会原子替换该文件，API 不返回本地模型、数据或数据库的绝对路径。

`ADMIN_TOKEN` 配置后，除 `/api/auth/*` 外的全部 API 和 WebSocket 都需要登录；登录交换为同源、`HttpOnly`、`SameSite=Strict` 的短期 Cookie，令牌不会存入浏览器本地存储或出现在 WebSocket URL 中。未配置 `ADMIN_TOKEN` 时服务为兼容旧部署而保持开放，`/api/auth/status` 会报告 `auth_required: false`，不得将此状态暴露到不可信网络。`AUTH_COOKIE_SECURE` 在 Nginx 终止 HTTPS 后必须设为 `true`；当前纯 HTTP 局域网部署保持 `false`，因此不应跨不受信任网络使用。

`API_UPSTREAM` 和 `NGINX_CLIENT_MAX_BODY_SIZE` 会在 Nginx 容器启动时渲染；默认值分别是 `host.docker.internal:8000` 和 `500m`。后端仍原生运行于 Mac mini，不加入 Compose。后端还会在读取请求时执行 `TRANSCRIBE_MAX_UPLOAD_BYTES`（默认 25MiB）、`MEETING_MAX_UPLOAD_BYTES`（默认 500MiB）和 `WS_MAX_AUDIO_BYTES`（默认 25MiB）限制；代理限制不得高于对应后端上限。实时 ASR 由一个固定线程 worker 串行执行，`ASR_WORKER_QUEUE_SIZE` 满时请求返回 503；`ASR_WORKER_REQUEST_TIMEOUT_SECONDS` 仅取消排队/结果回填，不能抢占已进入 MLX 的同步推理。会议与训练子进程会登记到 `TASK_STORE_PATH`（默认 `~/.astra/tasks.sqlite3`）。`MEETING_MAX_CONCURRENT_JOBS` 和 `TRAINING_MAX_CONCURRENT_JOBS`（均默认 1）在 SQLite 事务中全局预留，跨 API 重启和多 worker 生效，额度已满时接口返回 429。会议可由管理界面取消，系统会向其独立进程组发送 `SIGTERM`；训练仅终止其子进程 PID。状态查询及 API 启动会按 `MEETING_TASK_TIMEOUT_SECONDS`（默认 7200）和 `TRAINING_TASK_TIMEOUT_SECONDS`（默认 43200）终止超时任务并将其标记为 failed。启动收敛后还会根据 `MEETING_ARTIFACT_RETENTION_DAYS`（默认 30）及 `MEETING_ARTIFACT_MAX_BYTES`（默认 20GiB）清理会议根目录内**已登记且终态**的最旧任务产物；运行中、未登记及根目录外路径绝不删除。每次成功删除会写入 `tasks.sqlite3` 的 `artifact_cleanup_events`，并记录含 task ID、原因和回收字节数的应用日志。请在启用前按合规要求确认保留期并备份需要长期保存的会议数据。

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
docker compose exec frontend-nginx nginx -t
curl -fsS http://127.0.0.1:8080/
curl -fsS http://127.0.0.1:8080/api/health
curl -fsS http://127.0.0.1:8000/api/health
```
