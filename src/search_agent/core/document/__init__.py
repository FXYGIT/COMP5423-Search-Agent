"""固定文档集合的数据结构和访问接口。"""

from .reader import DocumentReader
from .schemas import (
    Document,
    DocumentPage,
    EvidencePassage,
    FindResult,
    InspectionResult,
    TextMatch,
)
from .store import DocumentStore

__all__ = [
    "Document",
    "DocumentPage",
    "FindResult",
    "TextMatch",
    "DocumentStore",
    "DocumentReader",
    "EvidencePassage",
    "InspectionResult",
]
