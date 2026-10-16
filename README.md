# COMP5423-Search-Agent

固定文档集合搜索 Agent 的最小开发骨架，基于 LangChain。

## 运行

```bash
uv sync --locked
uv run pre-commit install
```

复制 `.env.example` 为 `.env`，填写：

```dotenv
LLM__API_KEY=你的密钥
LLM__BASE_URL=服务商的兼容 API 地址
LLM__MODEL=服务商提供的 DeepSeek-V4-Flash 模型 ID
```

启动多轮流式聊天，或只问一个问题：

```bash
uv run search-agent
uv run python -m search_agent.cli
uv run search-agent "你好，请介绍一下你能使用的工具"
```

- 回答逐段显示，工具名称和参数在调用前显示，结果返回后立即显示。
- 连续输入可以追问，前面的模型和工具消息会带入下一轮。
- `/clear` 清空历史；`exit`、`quit`、`/exit` 退出；Ctrl+C 结束交互。
- 当前工具返回 `not_implemented`，用于检查调用流程，不产生搜索证据。

## 当前实现范围

**已实现**：模型配置、agent 组装、工具注册、多轮对话及流式 CLI。

**只保留注释和接口**：文档存储、BM25 检索、运行时技能加载、
搜索状态/上下文管理、trajectory 和 token 统计。业务方法使用
`NotImplementedError`，未注册的 middleware 不参与运行。

因此 CLI 可以聊天并展示工具调用，但真实语料搜索和完整作业功能需要后续补齐。
类和函数的职责、输入输出及 TODO 都在代码注释中，
详见 [Core 与 CLI 说明](docs/2-agent-middleware.md) 和
[检索实现说明](docs/1-retrieval-plan.md)。

## 网关复用

```python
from search_agent.core import create_search_agent

agent = create_search_agent()
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": question}],
})
answer = result["messages"][-1].text
# 连续对话时，下一次传入 result["messages"] 加上新的用户消息。
```

core 不负责输入输出或历史存储，CLI/网关各自维护会话消息。
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
