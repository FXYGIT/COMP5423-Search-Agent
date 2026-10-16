"""固定文档集合的数据结构和访问接口。"""

from .schemas import Document, DocumentPage, FindResult, TextMatch
from .store import DocumentStore

__all__ = ["Document", "DocumentPage", "FindResult", "TextMatch", "DocumentStore"]
