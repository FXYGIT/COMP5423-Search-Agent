"""各模块共享的状态协议；CLI/网关跨轮保存整个 state，而非仅 messages。"""

from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware.todo import Todo

from search_agent.core.document.schemas import EvidencePassage
from search_agent.core.trajectory import TrajectoryEntry


class SearchState(AgentState, total=False):
    """在 LangChain messages 之外保存搜索进度，避免只靠原始聊天历史。

    visited_doc_ids：已经阅读的原始文档 ID。
    selected_evidence：选择保留的证据，每项含 doc_id、text、source_url。
    active_subqueries / candidate_answer：当前子问题和待验证答案。
    active_skill：本轮使用的技能名称。
    trajectory：与提交示例一致的 role 消息；与模型请求裁剪视图分开保存。
    trajectory_message_count：已记录的原始消息数，避免跨轮重复追加。
    tool_calls / total_tokens：累计真实调用统计；缺少 API usage 时 total_tokens=None。
    summary/summarized_message_count：累计摘要和已经覆盖的原始消息数量。
    search_message_count：进度提取游标，仅处理新增消息，与轨迹和摘要游标独立。
    todos：LangChain 原生任务列表；未完成项映射为 active_subqueries。
    messages 始终保留完整原始过程；摘要和工具配对只改变模型请求视图。
    """

    visited_doc_ids: list[str]
    selected_evidence: list[EvidencePassage | dict[str, Any]]
    active_subqueries: list[str]
    candidate_answer: str | None
    search_message_count: int
    active_skill: str | None
    skill_instructions: str
    summary: str
    summarized_message_count: int
    todos: list[Todo]
    trajectory: list[TrajectoryEntry]
    trajectory_message_count: int
    tool_calls: int
    total_tokens: int | None
