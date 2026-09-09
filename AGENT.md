# AGENT.md

本文件是 Astra 仓库内代码代理和协作者的工作约定。

## 工作目录

- 先阅读 `README.md`、相关模块文档和当前 `git status`，再修改代码。
- 保留用户已有的未提交改动，不使用 `git reset --hard` 或 `git checkout --` 覆盖它们。
- 修改范围保持在当前任务需要的文件内；不要顺手重构无关模块。

## 项目约定

- 后端使用 Python 3.11、FastAPI 和类型标注，测试命令为 `PYTHONPATH=backend .venv/bin/pytest -q backend/tests`。
- MLX、Piper 和真实模型只在 Apple Silicon Mac mini 原生运行；不要把后端强行塞进 Linux Compose 容器。
- API 层负责协议和任务状态，模型编排放在 `backend/app/core/`，外部模型适配放在 `backend/app/models/`。
- 会议处理必须保留时间戳和说话人信息；默认只应用确定性纠错规则，不擅自开启 LLM 清洗。
- 一次性实验脚本不要放在根目录。可重复的会议回归脚本放在 `scripts/meeting/`，正式单元测试放在 `backend/tests/`。

## 录音和敏感数据

- 所有本地录音统一放在 `recordings/`，真实音频由 `.gitignore` 忽略，只保留 `.gitkeep` 占位文件。
- 模型权重放在 `models/`，会议产物放在 `~/Astra/meetings/`，均不提交到仓库。
- 不读取、输出或提交 `.env` 中的密钥；示例配置只使用 `.env.example`。

## 修改后验证

根据改动范围执行后端测试、Python 编译检查和前端构建。报告验证结果时明确区分单元测试、静态检查和真实模型/远端服务验收，不把其中一种当作全部验证。
