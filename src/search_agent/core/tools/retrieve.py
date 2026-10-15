from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from search_agent.core.retrieval.retriever import RetrievalMethod, get_retriever


class RetrieveInput(BaseModel):
    query: str = Field(
        min_length=1,
        description="搜索查询，可以包含实体名称、关键词或需要满足的条件。",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="最多返回的不同文档数量。",
    )
    method: RetrievalMethod = Field(
        default="hybrid",
        description=(
            "检索方法：sparse 为稀疏检索，"
            "dense 为 encoder 向量检索，"
            "instruction_dense 为指令感知向量检索，"
            "hybrid 为多种检索方法融合。"
        ),
    )


@tool(args_schema=RetrieveInput)
def retrieve(
    query: str,
    top_k: int = 5,
    method: RetrievalMethod = "hybrid",
) -> dict[str, Any]:
    """搜索固定文档集合，返回按相关性排序的文档 ID、分数和文本片段。

    用于发现与问题相关的文档。
    如需阅读全文或验证片段中的信息，使用 open_document。
    搜索结果不足时，可调整查询后再次调用。
    """
    query = query.strip()
    if not query:
        raise ToolException("搜索查询不能为空。")

    retriever = get_retriever()

    results = retriever.retrieve(
        query=query,
        top_k=top_k,
        method=method,
    )

    # 约定 results 为 list[RetrievalResult]，
    return {
        "query": query,
        "method": method,
        "results": [result.model_dump(mode="json") for result in results],
    }
