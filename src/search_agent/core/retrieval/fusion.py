"""多路排序融合骨架；融合不向 Agent 暴露独立工具。"""

from .schemas import RetrievalResult


def reciprocal_rank_fusion(
    rankings: list[list[RetrievalResult]],
    top_k: int | None = None,
    rank_constant: int | None = None,
) -> list[RetrievalResult]:
    """输入各检索器的有序结果，输出按融合分数排序的不同文档。

    TODO：按原始 doc_id 合并，名次从 1 开始，累加 1/(rank_constant + rank)。
    不直接相加不同检索器的原始分数。保留最合适的片段和来源，取 top_k。
    缺失于某一路的文档在该路贡献 0；融合分数写入 RetrievalResult.score。
    未提供参数时使用 AppConfig.retrieval.top_k / rrf_constant。
    """
    raise NotImplementedError
