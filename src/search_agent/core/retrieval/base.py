from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from search_agent.core.retrieval.schemas import RetrievalResult


class BaseRetriever(ABC):
    @abstractmethod
    def build_index(self) -> None:
        """读取语料并建立索引。"""
        ...

    @abstractmethod
    def save_index(self, path: Path) -> None: ...

    @abstractmethod
    def load_index(self, path: Path) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """返回按相关性降序排列的文档结果。"""
        ...
