# COMP5423-Search-Agent
COMP5423 小组作业 一个技能增强型搜索智能体，利用固定文档集合回答具有挑战性的问题。系统应将检索、大语言模型（LLM） 推理、工具使用和可复用技能整合为端到端工作流程


## Step 1
拉取项目后执行

1. 项目初始化

同步依赖
```python
uv sync --locked
uv run pre-commit install
```

下载数据集：BrowseComp-Plus 检索文档和评估集
```python
python ./scripts/download_browsecomp_plus_val.py 
python ./scripts/download_browsecomp_plus_corpus.py
```

## Step 2
执行cli对话

```python
python -m search_agent.core.app
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
