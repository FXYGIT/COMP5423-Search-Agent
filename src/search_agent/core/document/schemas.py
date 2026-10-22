from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Document(BaseModel):
    """固定语料中的完整文档，doc_id 是检索结果与原文访问之间的关联键。"""

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
    text 保存本次查找词句，matches 提供原文位置和上下文。
    """

    doc_id: str
    text: str
    matches: list[TextMatch]


class EvidencePassage(BaseModel):
    """可引用的原文证据，偏移始终指向固定语料原文，text 不使用生成式改写。"""

    doc_id: str
    text: str
    start: int
    end: int
    source_url: str | None = None


class InspectionResult(BaseModel):
    """文档内定位结果；method 表示采用的定位方式，而非全库检索方法。

    next_start 允许低置信度时继续顺序阅读，passages 为空仅代表已执行但无匹配。
    """

    doc_id: str
    query: str
    method: Literal["exact", "normalized", "passage_rank", "sequential"]
    passages: list[EvidencePassage]
    next_start: int | None = None
