"""多路排序融合骨架；融合不向 Agent 暴露独立工具。"""

from search_agent.core.config import get_config

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
    config = get_config()
    limit = top_k if top_k is not None else config.retrieval.top_k
    constant = rank_constant if rank_constant is not None else config.retrieval.rrf_constant
    if limit <= 0:
        return []

    scores: dict[str, float] = {}
    best: dict[str, RetrievalResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.doc_id] = scores.get(result.doc_id, 0.0) + 1.0 / (constant + rank)
            current = best.get(result.doc_id)
            # 同一个文档可能由不同 chunk/passage 命中，保留分数最高、片段最完整的那个。
            if current is None or result.score > current.score:
                best[result.doc_id] = result

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [best[doc_id].model_copy(update={"score": score}) for doc_id, score in ordered[:limit]]
