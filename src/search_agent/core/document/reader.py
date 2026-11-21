"""固定文档内的阅读优化与回退骨架，不做外网搜索。"""

import re

from .schemas import Document, EvidencePassage, InspectionResult
from .store import DocumentStore

# 段落分隔：空行（允许行内空白）。
_PARAGRAPH = re.compile(r"(?:\r?\n)[ \t\r]*(?:\r?\n)+")
# 与检索一致的词项切分，用于段落相关性打分。
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
# 句子边界，仅在段落过长时用于二次切分。
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# 单个段落超过该长度就按句子再切，避免一次返回过长片段。
_MAX_SEGMENT_CHARS = 1200
# 相邻命中间隔小于该值时合并，避免返回碎片化片段。
_MERGE_GAP = 40
_MAX_PASSAGES = 3


class DocumentReader:
    """在 DocumentStore 原文读取之上，定位与问题相关的证据片段。

    优化：查询相关段落选择、相邻命中合并、按字符预算限制返回长度。
    回退：精确查找 → 归一化查找 → 文档内段落排序 → 分页阅读。
    TODO：归一化必须保留到原文偏移的映射；找不到证据时不生成假片段。
    """

    def __init__(self, store: DocumentStore) -> None:
        """输入共用的语料访问层，避免重新加载整库。"""
        self.store = store

    def inspect(self, doc_id: str, query: str, max_chars: int | None = None) -> InspectionResult:
        """输入原始文档 ID、定位问题和长度预算，返回片段、定位方式及续读位置。

        TODO：在单篇原文内依次尝试定位策略；使用 EvidencePassage 返回真实偏移。
        兜底只打开同一固定语料文档，不访问 source_url 获取额外内容。
        max_chars 默认读取 self.store.config.max_chars。
        """
        budget = max_chars if max_chars is not None else self.store.config.max_chars
        if budget <= 0:
            raise ValueError("max_chars must be positive")

        document = self.store.get(doc_id)
        text = document.text
        if not text.strip():
            return InspectionResult(
                doc_id=doc_id, query=query, method="sequential", passages=[], next_start=None
            )

        start, end = self._locate_exact(text, query)
        if start >= 0:
            passages = self._span_passages(document, start, end, budget)
            return self._result(doc_id, query, "exact", passages, len(text))

        start, end = self._locate_normalized(text, query)
        if start >= 0:
            passages = self._span_passages(document, start, end, budget)
            return self._result(doc_id, query, "normalized", passages, len(text))

        passages = self._rank_passages(document, query, budget)
        if passages:
            return self._result(doc_id, query, "passage_rank", passages, len(text))

        passages = self._sequential_page(document, budget)
        return self._result(doc_id, query, "sequential", passages, len(text))

    # ---- 实现细节：四级回退策略与证据片段组装 ----

    @staticmethod
    def _result(
        doc_id: str,
        query: str,
        method: str,
        passages: list[EvidencePassage],
        total_chars: int,
    ) -> InspectionResult:
        """组装结果；next_start 指向最后一段之后，便于继续顺序阅读。"""
        next_start = passages[-1].end if passages and passages[-1].end < total_chars else None
        return InspectionResult(
            doc_id=doc_id,
            query=query,
            method=method,
            passages=passages,
            next_start=next_start,
        )

    @staticmethod
    def _locate_exact(text: str, query: str) -> tuple[int, int]:
        """第一级：原文里直接查找 query；找不到返回 (-1, -1)。"""
        needle = query.strip()
        if not needle:
            return -1, -1
        index = text.find(needle)
        if index < 0:
            return -1, -1
        return index, index + len(needle)

    @staticmethod
    def _locate_normalized(text: str, query: str) -> tuple[int, int]:
        """第二级：小写化并折叠空白后再查找，最后映射回原文偏移。"""
        needle = query.strip()
        if not needle:
            return -1, -1
        norm_text, offsets = _normalize_with_map(text)
        norm_query, _ = _normalize_with_map(needle)
        norm_query = norm_query.strip()
        if not norm_query:
            return -1, -1
        index = norm_text.find(norm_query)
        if index < 0:
            return -1, -1
        start = offsets[index]
        end = offsets[index + len(norm_query) - 1] + 1
        return start, end

    @staticmethod
    def _span_passages(
        document: Document, start: int, end: int, budget: int
    ) -> list[EvidencePassage]:
        """把命中位置向两侧扩展到上下文字符预算，返回一个片段。"""
        text = document.text
        context = max(0, (budget - (end - start)) // 2)
        left = max(0, start - context)
        right = min(len(text), end + context)
        return [
            EvidencePassage(
                doc_id=document.doc_id,
                text=text[left:right],
                start=left,
                end=right,
                source_url=document.source_url,
            )
        ]

    @staticmethod
    def _rank_passages(document: Document, query: str, budget: int) -> list[EvidencePassage]:
        """第三级：按 query 词命中数给段落打分，选前几段并合并相邻片段。"""
        text = document.text
        terms = {token.lower() for token in _TOKEN.findall(query)}
        if not terms:
            return []

        scored: list[tuple[int, int, int]] = []
        for seg_start, seg_end in _segments(text):
            segment_terms = {token.lower() for token in _TOKEN.findall(text[seg_start:seg_end])}
            hits = len(terms & segment_terms)
            if hits:
                scored.append((hits, seg_start, seg_end))
        if not scored:
            return []

        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen: list[tuple[int, int]] = []
        used = 0
        for _hits, seg_start, seg_end in scored:
            length = seg_end - seg_start
            if length > budget:
                seg_end = seg_start + budget
                length = budget
            if chosen and used + length > budget:
                continue
            chosen.append((seg_start, seg_end))
            used += length
            if used >= budget or len(chosen) >= _MAX_PASSAGES:
                break
        if not chosen:
            return []

        chosen.sort()
        return _to_passages(text, _merge_spans(chosen, budget), document)

    @staticmethod
    def _sequential_page(document: Document, budget: int) -> list[EvidencePassage]:
        """第四级：定位全失败时返回开头一段，由 next_start 支持继续阅读。"""
        text = document.text
        end = min(len(text), budget)
        return [
            EvidencePassage(
                doc_id=document.doc_id,
                text=text[:end],
                start=0,
                end=end,
                source_url=document.source_url,
            )
        ]


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """小写并折叠连续空白，同时记录每个归一化字符对应的原文字符下标。

    逐字符处理保证长度可控：若某个字符的小写形式不是单字符，则保留原字符，
    避免映射错位。空白折叠为单个空格，映射到该空白串的第一个字符。
    """
    out: list[str] = []
    offsets: list[int] = []
    in_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if not in_space:
                out.append(" ")
                offsets.append(index)
                in_space = True
            continue
        in_space = False
        lowered = char.lower()
        if len(lowered) != 1:
            lowered = char
        out.append(lowered)
        offsets.append(index)
    return "".join(out), offsets


def _segments(text: str) -> list[tuple[int, int]]:
    """把原文切成段落；过长段落再按句子切成不超过 _MAX_SEGMENT_CHARS 的块。"""
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in _PARAGRAPH.finditer(text):
        _append_segments(text, start, separator.start(), spans)
        start = separator.end()
    _append_segments(text, start, len(text), spans)
    return spans


def _append_segments(text: str, start: int, end: int, out: list[tuple[int, int]]) -> None:
    """去掉首尾空白后写入段落；超过长度上限时按句子边界切分。"""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return
    if end - start <= _MAX_SEGMENT_CHARS:
        out.append((start, end))
        return

    sentences: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE.finditer(text, start, end):
        if match.start() > cursor:
            sentences.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        sentences.append((cursor, end))
    if not sentences:
        out.append((start, end))
        return

    current_start, current_end = sentences[0]
    for next_start, next_end in sentences[1:]:
        if next_end - current_start <= _MAX_SEGMENT_CHARS:
            current_end = next_end
        else:
            out.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    out.append((current_start, current_end))


def _merge_spans(spans: list[tuple[int, int]], budget: int) -> list[tuple[int, int]]:
    """合并相邻或重叠的片段，并保证总长度不超过 budget。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= _MERGE_GAP:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    trimmed: list[tuple[int, int]] = []
    used = 0
    for start, end in merged:
        remaining = budget - used
        if remaining <= 0:
            break
        if end - start > remaining:
            end = start + remaining
        trimmed.append((start, end))
        used += end - start
    return trimmed


def _to_passages(
    text: str, spans: list[tuple[int, int]], document: Document
) -> list[EvidencePassage]:
    """把字符区间转成证据片段；text 必须是原文的精确切片。"""
    return [
        EvidencePassage(
            doc_id=document.doc_id,
            text=text[start:end],
            start=start,
            end=end,
            source_url=document.source_url,
        )
        for start, end in spans
        if end > start
    ]
