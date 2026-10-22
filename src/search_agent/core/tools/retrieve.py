"""Agent 唯一的检索工具入口，当前仅提供可调用的占位结果。"""

from typing import Any

from langchain.tools import tool

from search_agent.core.retrieval.schemas import RetrievalMethod


@tool
def retrieve(
    query: str,
    top_k: int = 5,
    method: RetrievalMethod = "hybrid",
    instruction: str = "Find evidence answering the query.",
) -> dict[str, Any]:
    """搜索固定文档集合，预期返回 doc_id、score、snippet 和 source_url。

    query 是搜索词句，top_k 是返回文档数量。
    method 选择 sparse/dense/instruction/hybrid，instruction 描述指令检索的意图。
    TODO：调用 Retriever.retrieve(query, top_k, method, instruction)。成功返回
    {status: "ok", result: list[RetrievalResult]}，当前 result=None，不能当作无命中。
    当前尚未实现搜索；not_implemented 不是空检索结果。
    """
    return {"status": "not_implemented", "message": "固定语料检索尚未实现。", "result": None}
