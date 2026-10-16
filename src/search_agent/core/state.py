"""作业结构化状态的接口草稿，待上下文与统计 middleware 实现后接入。"""

from typing import Any

from langchain.agents import AgentState


class SearchState(AgentState, total=False):
    """在 LangChain messages 之外保存搜索进度，避免只靠原始聊天历史。

    visited_doc_ids：已经阅读的原始文档 ID。
    selected_evidence：选择保留的证据，每项含 doc_id、text、source_url。
    active_subqueries / candidate_answer：当前子问题和待验证答案。
    active_skill：本轮使用的技能名称。
    trajectory / tool_calls / total_tokens：作业评估所需过程记录和统计。
    当前 MVP 只使用 LangChain 的 messages；以下字段尚未计算或填充。
    """

    visited_doc_ids: list[str]
    selected_evidence: list[dict[str, Any]]
    active_subqueries: list[str]
    candidate_answer: str | None
    active_skill: str | None
    trajectory: list[dict[str, Any]]
    tool_calls: int
    total_tokens: int
