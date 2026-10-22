"""任务规划直接复用 LangChain 的 TodoListMiddleware，不再维护另一套计划器。

输入：模型通过 write_todos 提交完整列表，每项 content 和 status。
输出：state.todos 更新，并产生真实 ToolMessage；状态为 pending/in_progress/completed。
当前即可运行，CLI 按 state.todos 展示进度。是否规划及如何拆解由主模型决定。
TODO：多线索技能实现后指导模型按“子问题→证据→验证”更新任务，避免独立规划 LLM。
"""

from langchain.agents.middleware import TodoListMiddleware

__all__ = ["TodoListMiddleware"]
