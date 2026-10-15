from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from search_agent.core.document.store import get_document_store


class OpenDocumentInput(BaseModel):
    doc_id: str = Field(
        min_length=1,
        description="检索结果返回的原始文档 ID。",
    )
    start: int = Field(
        default=0,
        ge=0,
        description="读取起点，使用原文字符偏移；从 0 开始。",
    )
    max_chars: int = Field(
        default=4000,
        ge=1,
        le=12000,
        description="本次最多读取的字符数量。",
    )


@tool(args_schema=OpenDocumentInput)
def open_document(
    doc_id: str,
    start: int = 0,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """读取指定文档的原文片段及来源信息。

    用于检查检索片段的上下文、阅读详细内容和验证候选答案。
    返回的 next_start 非空时，可以将其作为 start 继续阅读。
    所有位置均为原始文档中的字符偏移。
    """
    store = get_document_store()

    try:
        document = store.get(doc_id)
    except KeyError:
        raise ToolException(f"文档不存在：{doc_id}。请使用检索结果中的有效文档 ID。") from None

    total_chars = len(document.text)

    if start > total_chars:
        raise ToolException(f"start 超出文档长度，当前文档共 {total_chars} 个字符。")

    end = min(start + max_chars, total_chars)

    return {
        "doc_id": document.doc_id,
        "source_url": document.source_url,
        "text": document.text[start:end],
        "start": start,
        "end": end,
        "total_chars": total_chars,
        "next_start": end if end < total_chars else None,
    }
