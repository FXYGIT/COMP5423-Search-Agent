"""复用原生 TODO 规划，并提供显式的待验证候选记录工具。

输入：模型通过 write_todos 提交完整列表，每项 content 和 status。
输出：state.todos 更新，并产生真实 ToolMessage；状态为 pending/in_progress/completed。
当前即可运行，CLI 按 state.todos 展示进度。是否规划及如何拆解由主模型决定。
上下文 middleware 将未完成 TODO 映射到 active_subqueries，已完成项不再作为待办。
主模型负责按“子问题→证据→验证”更新任务，无需独立规划 LLM。
"""

from typing import Annotated

from langchain.agents.middleware import TodoListMiddleware as NativeTodoListMiddleware
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command


@tool
def record_candidate_answer(
    candidate_answer: str | None,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[str]:
    """Record an UNVERIFIED candidate answer while researching, or null to discard it.

    This only updates research memory; it does not establish facts or add evidence.
    Use document tools to verify the candidate before providing your final answer.
    """
    candidate = candidate_answer.strip() if candidate_answer else None
    return Command(
        update={
            "candidate_answer": candidate or None,
            "messages": [
                ToolMessage(
                    content="Unverified candidate recorded." if candidate else "Candidate cleared.",
                    name="record_candidate_answer",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


class TodoListMiddleware(NativeTodoListMiddleware):
    """保持 write_todos 原生语义；附加工具仅写 state，不发起规划模型调用。"""

    def __init__(self) -> None:
        super().__init__()
        self.tools = [*self.tools, record_candidate_answer]


__all__ = ["TodoListMiddleware"]
