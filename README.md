# ResolveAI · 电商 Agent 工作流平台

Resolv AI 是一个电商客服 Agent 学习与原型项目，包含 Python 后端、React 前端、会话持久化、工具审批和知识检索模块。

此公开版本提供源码、测试和模拟数据。业务知识文档、开发验收记录、本地数据库、模型权重和运行凭据不随仓库分发。完整业务流程需要自行配置模型、存储和业务适配器。

## 功能范围

- **对话与分流**：流式输出、意图理解、单一主诉求处理，以及待处理诉求队列。
- **工具调用**：订单与物流查询、需要显式审批的写操作、执行回执与重复提交保护。
- **会话恢复**：MySQL 会话记录、LangGraph checkpoint、历史摘要和上下文预算管理。
- **知识检索代码**：MySQL 权威正文、Milvus 混合检索、BGE 向量与重排、证据引用和不足时的兜底。
- **售后申请原型**：退款判断、申请入口、本地申请登记与回执；真实支付退款、审核后台和人工客服系统需要另行接入。
- **物流扩展**：MCP 工具与明确标记的模拟物流数据。
- **前端工作台**：会话列表、聊天面板、工具审批、申请卡片和运行信息查看。

默认生产适配器没有完整接入真实身份、订单、支付和物流服务。模拟执行或本地申请登记不代表真实业务已经完成。

## 技术组成

| 部分 | 主要技术 |
| --- | --- |
| 后端 | Python、FastAPI、LangChain、LangGraph、Pydantic |
| 存储 | SQLAlchemy、Alembic、MySQL、SQLite、Milvus |
| 检索 | BGE-M3、BGE reranker、PyTorch、Transformers |
| 前端 | React、TypeScript、Vite、Tailwind CSS |
| 依赖管理 | uv / uv.lock；npm / package-lock.json |

## 仓库结构

~~~text
.
├── src/resolv/       # API、图编排、业务服务、存储与工具
├── frontend/         # 前端源码、静态资源和测试
├── migrations/       # MySQL 数据结构迁移
├── scripts/          # 初始化、诊断、迁移和显式集成验证工具
├── tests/            # 单元测试、模拟数据和回归检查
├── logistics/        # 模拟物流数据
├── compose.mysql.yaml
├── compose.milvus.yaml
├── .env.example      # 无真实凭据的配置模板
├── pyproject.toml
└── uv.lock
~~~

## 安装依赖

使用 Python 3.12（见 .python-version）和 uv。先创建或复用项目内虚拟环境，再按锁文件安装：

~~~bash
uv venv
uv sync --frozen
~~~

项目包含 PyTorch 等依赖，安装后的环境可能较大；这些依赖不包含在 Git 仓库中。uv 的项目与锁文件说明见[官方文档](https://docs.astral.sh/uv/concepts/projects/layout/)。

前端使用与 frontend/package-lock.json 一致的 npm 依赖：

~~~bash
cd frontend
npm ci
npm run dev
~~~

前端默认在 http://127.0.0.1:5173 启动，并将 /api 请求代理到 http://127.0.0.1:8000。仅启动前端不会启动或初始化后端。

## 本地配置

在项目根目录创建本地配置：

~~~bash
cp .env.example .env
~~~

按自己的模型服务填写 OPENAI_*，并单独填写 SUMMARY_*。CONTEXT_* 中的模型窗口、tokenizer 和预算应与实际模型一致。数据库密码、连接串和 API 密钥只保存在本地配置或部署环境中。

本项目没有分发可直接复用的业务数据库。后端启动会检查既有表结构，不会自动迁移、播种数据或下载模型。完整 HTTP 服务需要先准备：

1. 与 migrations/ 对应的 MySQL 会话结构和 checkpoint 表。
2. SQLite 订单、运单、工具回执、退款申请与提交回执等业务表。
3. 模型与摘要服务，以及相应的上下文配置。
4. 使用知识问答时，自行准备允许使用的业务资料、知识数据库、Milvus 索引、本地模型和经过验证的发布选择配置。

scripts/ 中的迁移和验收工具包含目标身份、资源归属及前置条件校验。历史验收脚本依赖未公开的记录，不能直接作为新部署的一键安装流程。

完成上述准备后，后端入口为：

~~~bash
uv run --frozen uvicorn resolv.main:app --host 127.0.0.1 --port 8000
~~~

公开版没有 knowledge/ 目录中的资料和发布选择配置；缺少选择配置时知识分支保持未接入。不要通过绕过发布校验来启用未经验证的知识版本。

## 开发检查

以下后端检查使用模拟或临时数据，不需要真实业务文档：

~~~bash
uv run --frozen pytest -q tests/test_config.py tests/test_llm.py tests/test_knowledge.py tests/test_logistics_contracts.py
uv run --frozen python scripts/inspect_tools.py
~~~

前端检查在 frontend/ 下执行：

~~~bash
npm test
npm run typecheck
npm run build
npm run lint
~~~

后端完整测试集合保留了开发阶段的回归检查，其中部分会读取未公开的业务文档、验收记录或历史提交。上述定向检查不等于完整测试集合通过。需要完整复现时，应在自己的私有环境补齐相应材料；仓库不附带旧环境的验收结论。

以 smoke_ 开头的脚本并非全部离线：部分会调用模型、访问外部服务或操作指定测试数据库。运行前先阅读对应脚本和参数。

## 公开版本边界

- 不包含 knowledge/ 业务资料、测试夹具中的原业务政策副本、docs/ 开发文档、开发截图及本地设计检查记录。
- 不包含 .env、数据库、备份、运行日志、开发快照、虚拟环境、node_modules 或构建输出。
- 配置模板不包含真实密钥；作者信息只保留公开用户名。
- 本次公开发布从独立的首次提交开始，不携带原项目开发历史。
