"""结构化搜索状态与上下文预算的算法接口。"""

from typing import Any

from langchain_core.messages import AnyMessage

from search_agent.core.state import SearchState


def extract_search_progress(state: SearchState) -> dict[str, Any]:
    """输入完整状态，输出 visited_doc_ids/selected_evidence/active_subqueries 等更新。

    TODO：处理新工具结果，按 doc_id 和原文区间去重证据，保留候选答案及待验证线索。
    not_implemented 工具结果不可转成证据；用游标避免反复处理旧消息。
    """
    raise NotImplementedError


def select_context(state: SearchState, max_tokens: int = 16000) -> list[AnyMessage]:
    """输入原始状态和模型输入预算，输出选中的历史消息副本。

    TODO：预留 system/工具定义/输出空间，选择证据与近期完整轮次；超预算时摘要。
    只选择尚未被 summary 覆盖的消息，保留最新问题；完成后再次检查工具消息配对。
    不删除原始历史，不把字符数伪称 API token 用量。
    """
    raise NotImplementedError
