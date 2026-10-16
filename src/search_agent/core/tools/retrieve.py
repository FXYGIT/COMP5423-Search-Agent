"""Agent 唯一的检索工具入口，当前仅提供可调用的占位结果。"""

from typing import Any

from langchain.tools import tool


@tool
def retrieve(query: str, top_k: int = 5) -> dict[str, Any]:
    """搜索固定文档集合，预期返回 doc_id、score、snippet 和 source_url。

    query 是搜索词句，top_k 是返回文档数量。
    TODO：调用 Retriever.retrieve(query, top_k)，将结果序列化为字典列表。
    当前尚未实现搜索；not_implemented 不是空检索结果。
    """
    return {"status": "not_implemented", "message": "固定语料检索尚未实现。"}
