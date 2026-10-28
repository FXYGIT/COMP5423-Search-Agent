# COMP5423-Search-Agent

固定文档集合搜索 Agent 的可运行开发骨架，基于 LangChain；业务算法保留接口与 TODO。

## 运行

```bash
uv sync --locked
uv run pre-commit install
```

复制 `.env.example` 为 `.env`。所有模块统一通过 `core/config.py` 的 `AppConfig` 读取，
系统环境变量优先于 `.env`。真实聊天需要填写：

```dotenv
LLM__API_KEY=你的密钥
LLM__BASE_URL=服务商的兼容 API 地址
LLM__MODEL=服务商提供的 DeepSeek-V4-Flash 模型 ID
```

embedding、Chroma、检索、建库、文档、技能、上下文、评估和日志配置也都在 `.env` 中。
建库与查询共用 `EMBEDDING__DENSE_MODEL` / `EMBEDDING__INSTRUCTION_MODEL`，以及
`CHROMA__PATH` 和对应 collection。`INDEXING__CATEGORY` 决定本次构建哪类索引。
配置示例和默认值见 [.env.example](.env.example)。

Agent 和脚本统一使用 root logger，默认仅写 `logs/search_agent.log`，
由 `LOGGING__PATH`、`LOGGING__LEVEL` 控制。日志格式为时间、级别、模块、消息；
不添加控制台日志 handler，CLI 的 Rich/流式交互照常显示。

启动多轮流式聊天，或只问一个问题：

```bash
uv run search-agent
uv run python -m search_agent.cli
uv run search-agent "你好，请介绍一下你能使用的工具"
uv run search-agent --plain --verbose
```

- 回答逐段显示，工具名称和参数在调用前显示，结果返回后立即显示。
- 连续输入可以追问，前面的模型和工具消息会带入下一轮。
- Rich 面板显示工具调用、任务计划、当前技能和支持证据；长工具结果默认截短预览。
- `/help` 查看命令；`/todos`、`/skills`、`/evidence`、`/status` 查看对应状态。
- `/clear` 清空完整状态；`exit`、`quit`、`/exit` 退出；生成中 Ctrl+C 中断本轮。
- `--plain` 使用纯文本；`--verbose` 展示完整工具结果。
- 检索、文档、技能工具返回 `not_implemented`，不产生假证据；`write_todos` 和
  `record_candidate_answer` 可实际使用。

## 当前实现范围

**已实现**：模型配置、agent 组装、工具注册、多轮完整状态保存、流式 CLI、
原生 TODO 规划、候选答案记录、成功文档结果的搜索状态提取、原文证据去重、
请求预算选择、增量摘要及工具消息配对。原始历史完整保留；主模型和摘要模型的
消息轨迹、工具调用次数和实际 API usage 使用同一统计协议。

**只保留注释和接口**：文档存储、相关片段选择与定位回退；BM25、稠密、指令检索和
RRF 融合；技能发现、正文/资源加载与脚本执行；技能辅助模型的轨迹接入。
这些业务方法使用 `NotImplementedError`；对应 middleware 已注册，未实现部分透传。
上下文模块可消费约定的成功工具结果，真实语料/技能联调仍依赖 A、C 的实现。

因此 CLI 可以聊天并展示工具调用，但真实语料搜索和完整作业功能需要后续补齐。
类和函数的职责、输入输出及 TODO 都在代码注释中，
详见 [Core 与 CLI 说明](docs/2-agent-middleware.md) 和
[检索实现说明](docs/1-retrieval-plan.md)、[扩展骨架说明](docs/4-extension-scaffold.md) 和
[Agent 与上下文管理说明](docs/6-agent-context.md)。

`CONTEXT__MAX_TOKENS` 包含 system、工具定义与输出预留，
`CONTEXT__SUMMARY_TRIGGER_TOKENS` 控制历史摘要阈值；
`CONTEXT__SUMMARY_ENABLED=false` 可关闭额外摘要调用，预算裁剪仍生效。
摘要默认使用同一个 DeepSeek-V4-Flash 模型，也会产生真实 API 费用。
预算按本地文本估算控制，并非 DeepSeek tokenizer 的精确上限；统计只使用 API usage。

## 网关复用

```python
from search_agent.core import create_search_agent
from langchain_core.messages import HumanMessage

agent = create_search_agent()
state = {"messages": []}  # 每个会话单独维护
state["messages"].append(HumanMessage(content=question))
state = await agent.ainvoke(state)
answer = state["messages"][-1].text
# 连续对话时保留完整 state，再向 messages 追加新 HumanMessage。
```

core 不负责终端输入输出或持久化，CLI/网关各自维护完整会话状态。
未使用数据库、checkpointer 或额外会话管理器。

## 数据准备与检查

聊天 MVP 无需下载语料；实现检索时再运行：

```bash
uv run python scripts/download_browsecomp_plus_corpus.py
uv run python scripts/download_browsecomp_plus_val.py
```

离线测试不调用真实 API：

```bash
uv run pytest
uv run ruff check .
uv run mypy .
```

## 评估骨架

离线建库默认写入当前目录的 `./chroma`。评估入口读取本地验证集，每题使用独立状态，
生成 `test_prediction.jsonl` 和 `evaluation_report.json/md`。

```bash
uv run python scripts/build_vector_db.py
uv run python scripts/run_predictions.py
uv run python scripts/evaluate_predictions.py
```

数据读取、JSONL 读写、normalized EM、用量汇总和报告输出已实现。
轨迹 schema 对齐根目录 `test_prediction.jsonl` 示例：system/user/assistant/tool 消息、
JSON 字符串工具参数、每次 assistant 的 usage。主 Agent 已记录实际调用；
流式模型启用 `stream_usage=True`，若服务端未返回 usage 则聊天继续、评估拒绝导出。
**建库、embedding、Chroma 工具适配及技能辅助 LLM 轨迹接入仍为 TODO**，前两条命令会明确提示
尚未接通，不会生成假预测。脚本只写日志，失败以非零状态退出；原因见日志文件。
已有预测文件可以直接复评。评估路径、split、题数限制、输出文件均用 `EVALUATION__*` 配置。
参考答案只用于评分；验证集文件不等于正式测试集提交。
接口和参数见 [评估模块说明](docs/5-evaluation-scaffold.md)。

## 分支与合并协作流程

本项目采用受保护的 `main` 主分支和多个 `feature/xxx` 开发分支进行协作。

### 分支约定

- `main`：存放合并后的项目代码，禁止直接推送，所有修改必须通过 Pull Request（简称 PR）合入。
- `feature/xxx`：用于开发具体功能，例如 `feature/login`、`feature/user-profile`。每个功能尽量使用独立分支。

基本流程：

最新 main → 创建 feature/xxx → 开发并推送 → 提交 PR → 指派负责人 → 确认后合并到 main

### 1. 开始开发

先更新本地 main，再创建自己的功能分支：
```bash
git checkout main
git pull --ff-only origin main
git checkout -b feature/功能名称
```
### 2. 提交并推送代码

完成修改后，在功能分支上提交并推送：
```bash
git add <需要提交的文件>
git commit -m "说明本次修改内容"
git push -u origin feature/功能名称
```
提交前请检查改动，并完成相关功能验证。

### 3. 创建 PR

在 GitHub 上创建 Pull Request：

- **base（目标分支）**：选择 `main`。
- **compare（来源分支）**：选择自己的 `feature/xxx`。
- **标题**：简要说明本次修改。
- **描述**：写清修改内容、验证方式，以及需要特别注意的事项。
- **Assignees（负责人）**：选择 @你的GitHub用户名，由项目负责人跟进并确认合并。
- 如需负责人审核，可同时在 **Reviewers（审核人）** 中选择该账号；自己创建的 PR 无法将自己设为审核人。

如果功能尚未完成，请先创建 Draft PR，完成后再标记为 Ready for review。

### 4. 确认与合并

- 提交 PR 后，由项目负责人检查改动并安排合并，请勿自行合并。
- 如果需要调整，继续向原功能分支提交并推送即可，PR 会自动更新，无需重新创建。
- 合并完成后，可以删除对应的远程功能分支。
- 开始下一项功能前，重新更新 main，并创建新的功能分支。
