# Astra 项目评审与改进路线图

> 评审时间：2026-09-10
>
> 范围：FastAPI 后端、React/Vite 前端、Nginx Compose 部署、会议/训练任务和测试工程化
>
> 基线：`main` 比 `origin/main` 领先 2 个提交；评审时工作区已有未提交业务改动，本报告不将它们纳入本次实现范围。

## 结论

Astra 已完成从实时语音到会议纪要、说话人声纹、人工审校和 Qwen3-ASR 训练候选数据导出的 MVP 闭环。后端单元测试和前端生产构建均可通过。下一阶段的主风险不在单点功能，而在局域网访问控制、音频/模型资源隔离、长任务治理、数据保留和持续交付能力。

本路线图按依赖关系拆分；每项任务应独立提交，避免把鉴权、任务系统、前端重构等大改动混入同一个 commit。

## 已验证基线

- 后端：`PYTHONPATH=backend .venv/bin/pytest -q backend/tests`，76 passed。
- 前端：`npm --prefix frontend run build`，TypeScript 检查和 Vite 生产构建通过。
- 部署：`docker compose config --quiet` 通过。
- 以上仅是自动化基线；未执行真实 Mac mini 模型、远端 LLM 或浏览器端到端验收。

## 风险清单

| 优先级 | 问题 | 影响 | 证据位置 |
|---|---|---|---|
| P0 | 管理、声纹、会议、训练和 WebSocket 接口没有认证/授权 | 局域网任意访问者可读取录音/声纹、修改 LLM 配置、触发训练和消耗模型资源 | `backend/app/main.py`、`backend/app/api/` |
| P0 | 上传和 WebSocket 音频缓冲缺少一致的尺寸、时长和并发限制 | 内存耗尽、超长推理、拒绝服务 | `http_routes.py`、`meeting_routes.py`、`session_manager.py` |
| P0 | Nginx 上游固定为 `192.168.3.18:8001`，25MB 代理限制又与会议接口 500MB 限制冲突 | 部署漂移；大录音被代理提前拒绝 | `frontend/nginx.conf`、`deploy/README.md` |
| P0 | 实时 ASR 在 async 请求中同步调用 MLX | 推理期间可能阻塞 API 事件循环及其他会话 | `backend/app/models/asr_client.py` |
| P1 | 会议子进程和训练进程缺少持久状态、取消、配额、重启恢复和产物清理 | API 重启后任务不可管理；磁盘可无限增长 | `meeting_routes.py`、`core/training.py` |
| P1 | 审校 JSONL 采用全量读写且没有版本/锁 | 并发审校会丢失更新；任务大时延迟增加 | `meeting_routes.py` |
| P1 | 配置写入依赖启动时工作目录，接口还暴露本地绝对路径 | 配置写错位置；向客户端暴露运行环境细节 | `config.py`、`main.py` |
| P2 | 依赖存在范围版本和 SSH Git 依赖，缺少锁定/供应链检查 | 重建不可复现，部署机依赖 SSH 环境 | `backend/requirements.txt` |
| P2 | 无 CI、lint、类型检查、前端单测或浏览器 E2E | 回归只能依赖人工发现 | 仓库根目录、`frontend/` |
| P2 | `UploadPage.tsx` 职责过多，网络请求与 WebSocket 逻辑分散 | 难测试、难维护、重连和错误语义不一致 | `frontend/src/UploadPage.tsx` |
| P2 | 每个通知 WebSocket 每秒查询 SQLite | 客户端增多时产生无效轮询 | `notification_routes.py` |

## 开发任务拆分

### 阶段 0：部署契约与访问边界

| ID | 任务 | 前置 | 验收标准 | 建议 commit |
|---|---|---|---|---|
| P0-01 | 配置化 Nginx API 上游和上传限制 | 无 | `API_UPSTREAM`、`NGINX_CLIENT_MAX_BODY_SIZE` 写入 `.env.example`；Compose 渲染 Nginx 模板；默认值与 Mac mini 原生后端 `8000` 一致；`docker compose config` 和 Nginx 配置检查通过 | `chore(deploy): configure nginx upstream and upload limit` |
| P0-02 | 引入局域网管理员认证 | P0-01 | 所有 `/api/*` 管理接口和 WebSocket 需要令牌；静态前端保持可访问；未认证返回 401/WS 1008；不记录令牌 | `feat(auth): protect API and websocket endpoints` |
| P0-03 | 为上传、SSE 和 WS 加资源上限 | P0-02 | 文件按块落盘；限制请求体、解码后时长、WS 累积音频和每 IP 并发数；超限返回 413/明确 WS 错误 | `feat(api): enforce audio resource limits` |
| P0-04 | 将实时 ASR 隔离到受控推理 worker | P0-03 | API 进程不会因模型推理阻塞；单 worker 串行执行或有明确队列；排队、超时、取消和失败可观测 | `feat(asr): isolate realtime inference worker` |

**P0-02 的设计决策（实施前确认）**：确定是否仅信任内网、是否要多人账号、令牌存储方式、前端令牌注入方式，以及 Nginx 终止 TLS 还是仅用于反代。该任务会改变服务访问行为，不能与 P0-01 混合提交。

### 阶段 1：任务与数据治理

| ID | 任务 | 前置 | 验收标准 | 建议 commit |
|---|---|---|---|---|
| P1-01 | 持久化会议/训练任务状态 | P0-02 | SQLite 记录 task ID、PID、状态、时间、输入/输出；服务重启后能重新展示状态 | `feat(tasks): persist meeting and training jobs` |
| P1-02 | 增加任务取消、并发配额和超时回收 | P1-01 | 可安全终止子进程组；会议和训练有独立并发限制；超时标记失败且回收资源 | `feat(tasks): manage task lifecycle and quotas` |
| P1-03 | 制定并实现会议产物/样本保留策略 | P1-01 | 可配置保留期和最大磁盘占用；清理不删除运行中任务；产生日志/审计记录 | `feat(storage): retain and clean meeting artifacts` |
| P1-04 | 用 SQLite 管理审校状态，JSONL 仅用于导出 | P1-01 | 分段更新具备版本号或事务；并发编辑返回冲突；训练集可从已批准分段重建 | `feat(review): persist reviewed transcript segments` |
| P1-05 | 显式化运行时配置路径与敏感字段输出 | P0-02 | 使用 `ASTRA_CONFIG_PATH`；接口不返回绝对路径和敏感值；配置写入原子化 | `feat(config): make runtime config location explicit` |

### 阶段 2：质量、可维护性与可观测性

| ID | 任务 | 前置 | 验收标准 | 建议 commit |
|---|---|---|---|---|
| P2-01 | 固化 Python/Node 依赖与供应链检查 | 无 | 受控 lock/constraints；CI 可从干净环境安装；SSH Git 依赖有替代安装路径 | `build: lock application dependencies` |
| P2-02 | 建立 CI 和静态检查 | P2-01 | PR 运行 pytest、Ruff、类型检查、前端 build；失败阻止合并 | `ci: add quality gates` |
| P2-03 | 为关键 API 和前端工作流补测试 | P0-02、P1-04 | 覆盖认证、配额、任务恢复、审校冲突；Playwright 覆盖上传至审校主链路 | `test: cover protected meeting workflow` |
| P2-04 | 拆分前端页面与统一网络客户端 | P0-02 | `UploadPage` 按领域拆分；统一处理错误、取消、重试和 WS 生命周期；保留现有行为 | `refactor(frontend): modularize meeting workspace` |
| P2-05 | 事件驱动通知与运行指标 | P1-01 | 通知不再每连接轮询 SQLite；日志包含 task/session ID；提供任务、队列和磁盘指标 | `feat(obs): add task telemetry and notification hub` |

## 推荐执行顺序

1. 先完成 P0-01，消除已知部署契约漂移；这是本报告对应的最小实现任务。
2. 在确定访问模型后实施 P0-02，再实施 P0-03；避免先引入限制而仍暴露管理接口。
3. P0-04 与 P1-01 可以并行设计，但推理 worker 的队列状态应复用 P1-01 的任务存储模型。
4. P1-02、P1-03、P1-04 依次实施，保证长任务和录音数据可控。
5. P2-01/P2-02 可尽早并行落地，为后续重构提供回归保护。

## 每项任务的交付要求

- 一个任务一个逻辑 commit；不混入已有工作区变更。
- 新增接口、权限、限制或状态转换时，同时补充后端单元/集成测试。
- 更改 Compose、Nginx 或环境变量时，同时更新 `.env.example` 和 `deploy/README.md`。
- 验证至少包括受影响测试、前端构建（涉及前端时）和 `docker compose config --quiet`（涉及部署时）。
- 真实模型、远端 LLM、浏览器权限和 TLS 验收需在 Mac mini 目标环境单独执行并记录结果。
