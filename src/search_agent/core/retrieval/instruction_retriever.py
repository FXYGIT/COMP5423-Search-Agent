from __future__ import annotations

from search_agent.core.retrieval.base import BaseRetriever


class InstructionRetriever(BaseRetriever):
    def format_query(self, query: str) -> str:
        raise NotImplementedError

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def encode_query(self, query: str) -> list[float]:
        raise NotImplementedError

    # 实现 build_index / save_index / load_index / retrieve
