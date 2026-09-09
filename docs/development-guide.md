# Astra 开发指引

## 项目结构

```text
Astra/
├── backend/app/       FastAPI 应用、API、流水线和模型适配器
├── backend/tests/     后端单元测试
├── frontend/src/      React + Vite 前端
├── scripts/meeting/   会议 Workflow 回归和纪要工具
├── recordings/        本地录音样本，真实音频被 Git 忽略
├── deploy/            部署和 Mac mini 预检说明
├── docs/              架构、评审、用户和开发文档
├── models/             本地模型目录，不提交模型文件
└── docker-compose.yml 前端 Nginx Compose 配置
```

根目录不再放一次性测试脚本。新的可复用测试进入 `backend/tests/`；会议相关的可重复命令进入 `scripts/meeting/`。

## 开发环境

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
PYTHONPATH=backend .venv/bin/pytest -q backend/tests
```

后端依赖 MLX，必须在 Apple Silicon 环境验证真实模型行为。没有模型或远端 LLM 时，仍可运行不依赖模型的单元测试。

启动后端：

```bash
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动前端开发服务器：

```bash
cd frontend
npm install
npm run dev
```

## 核心边界

- `app/core/pipeline.py` 负责实时语音会话的编排；
- `app/core/workflow.py` 负责会议处理阶段和时间戳/说话人数据；
- `app/models/` 只负责 ASR、LLM、TTS、标点等外部模型适配；
- `app/api/` 负责 HTTP、WebSocket 和任务状态，不应承载模型业务逻辑；
- 会议任务由 `app/core/meeting_cli.py` 独立进程执行，API 进程只负责创建任务和轮询状态；
- `scripts/meeting/download_and_test_workflow.py` 是当前录音回归入口，不要恢复已删除的一次性根目录脚本。

新增模型引擎时，优先实现现有客户端契约，在 `app/main.py` 的构建函数中接入，再补配置和单元测试。会议转写应保持时间戳和说话人信息，不把普通对话场景的热词配置直接复用到会议 ASR。

## 配置和数据

从 `.env.example` 增加配置项，并在 `backend/app/config.py` 中提供类型化默认值。禁止提交真实 API key、模型权重、用户录音和会议产物。录音只放在固定目录 `recordings/`；会议运行产物放在 `~/Astra/meetings/`。

## 验证清单

提交前至少执行：

```bash
PYTHONPATH=backend .venv/bin/pytest -q backend/tests
.venv/bin/python -m py_compile scripts/meeting/download_and_test_workflow.py scripts/meeting/generate_llm_minutes.py
```

涉及前端时再执行：

```bash
cd frontend
npm run build
```

静态测试不能证明真实 MLX 推理、远端 LLM 连通性或完整会议任务成功；需要在具备模型和服务的 Mac mini 上单独做验收。
