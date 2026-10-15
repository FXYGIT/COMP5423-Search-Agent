from __future__ import annotations

from search_agent.core.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    # 实现 build_index / save_index / load_index / retrieve
