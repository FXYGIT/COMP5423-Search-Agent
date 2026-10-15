# COMP5423-Search-Agent
COMP5423 小组作业 一个技能增强型搜索智能体，利用固定文档集合回答具有挑战性的问题。系统应将检索、大语言模型（LLM） 推理、工具使用和可复用技能整合为端到端工作流程


## Step 1
拉取项目后执行

1. 同步依赖 

```python
uv sync --locked
uv run pre-commit install
```

## Step 2
执行cli对话

```python
python -m search_agent.core.app
```