from __future__ import annotations

from pydantic import BaseModel


class Document(BaseModel):
    """完整文档"""

    doc_id: str
    text: str
    source_url: str | None = None


class DocumentPage(BaseModel):
    """分段读取文档的返回结果。

    所有位置均为原文字符偏移，区间采用 [start, end)。
    next_start 为 None 表示已读到文档末尾。
    """

    doc_id: str
    source_url: str | None = None
    text: str
    start: int
    end: int
    total_chars: int
    next_start: int | None = None


class TextMatch(BaseModel):
    """文档内的一次文本匹配。

    start/end：匹配文本在原文中的位置。
    context_start/context_end：上下文在原文中的位置。
    """

    start: int
    end: int
    matched_text: str
    context_start: int
    context_end: int
    context: str


class FindResult(BaseModel):
    """文档内查找的返回结果。

    matches 为空表示未找到。
    truncated 表示还有匹配因数量限制而未返回。
    """

    doc_id: str
    text: str
    matches: list[TextMatch]
    truncated: bool = False
