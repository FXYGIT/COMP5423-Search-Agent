from __future__ import annotations

from typing import Literal

from search_agent.core.retrieval.base import BaseRetriever
from search_agent.core.retrieval.schemas import RetrievalResult

RetrievalMethod = Literal["sparse", "dense", "instruction_dense", "hybrid"]


class Retriever:
    def __init__(self, backends: dict[str, BaseRetriever]) -> None:
        self.backends = backends

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: RetrievalMethod = "hybrid",
    ) -> list[RetrievalResult]:
        """选择后端，必要时融合，最终返回 top_k 个不同文档。"""
        raise NotImplementedError

    def _fuse(
        self,
        rankings: list[list[RetrievalResult]],
    ) -> list[RetrievalResult]:
        """例如使用 RRF，避免直接相加不同后端的原始分数。"""
        raise NotImplementedError


def get_retriever() -> Retriever:
    """获取共享的文档存储实例。

    输入：
        无显式参数；语料路径由项目配置提供。

    输出：
        已初始化的 DocumentStore。

    预期行为：
        首次调用时初始化，后续调用复用实例，
        避免每次调用工具都重新加载语料。
    """
    raise NotImplementedError
