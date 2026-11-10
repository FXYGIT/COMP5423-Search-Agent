"""三类检索与融合的统一入口。"""

import logging
from pathlib import Path

import numpy as np

from search_agent.core.config import AppConfig, get_config

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .instruction import InstructionRetriever
from .schemas import RetrievalMethod, RetrievalResult

logger = logging.getLogger(__name__)


class Retriever:
    """统一协调 BM25、encoder dense、instruction dense 和融合。

    实现说明：hybrid 采用 **chunk 级分数融合**——BM25 保持 1000 词 chunk，
    稠密索引复用同一份 ``chunk_map``，于是三路分数可以在同一 chunk 下标上相加，
    再按文档取最大值。RRF 仍保留在 ``fusion.py``，作为可替换的融合方式。
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """共用 .env 配置快照，TODO：据此组装后端；构造阶段不加载模型。"""
        self.config = config or get_config()
        self._sparse: BM25Retriever | None = None
        self._sparse_loaded = False
        self._dense: DenseRetriever | None = None
        self._dense_loaded = False
        self._instruction: InstructionRetriever | None = None
        self._instruction_loaded = False
        self._chunk_index_cache: dict[tuple[str, int, int], int] | None = None

    def _get_sparse(self) -> BM25Retriever:
        """稀疏后端只在首次使用时创建，避免构造 Retriever 就加载索引。"""
        if self._sparse is None:
            self._sparse = BM25Retriever(self.config)
        return self._sparse

    def _get_dense(self) -> DenseRetriever:
        """稠密后端同样懒加载，模型只在第一次 load_index 时进显存。"""
        if self._dense is None:
            self._dense = DenseRetriever(self.config)
        return self._dense

    def _get_instruction(self) -> InstructionRetriever:
        """指令稠密后端懒加载，使用独立 collection。"""
        if self._instruction is None:
            self._instruction = InstructionRetriever(self.config)
        return self._instruction

    def _ensure_backends(self, method: RetrievalMethod) -> None:
        """按目标方法加载所需后端；未实现的方法给出明确错误。"""
        if method in ("sparse", "hybrid") and not self._sparse_loaded:
            self._get_sparse().load_index()
            self._sparse_loaded = True
        if method == "dense" and not self._dense_loaded:
            self._get_dense().load_index()
            self._dense_loaded = True
        if method == "instruction" and not self._instruction_loaded:
            self._get_instruction().load_index()
            self._instruction_loaded = True
        if method == "hybrid":
            # 三路融合：稀疏必选，稠密/指令缺失时自动降级。
            if not self._instruction_loaded:
                try:
                    self._get_instruction().load_index()
                    self._instruction_loaded = True
                except FileNotFoundError as error:
                    logger.warning("instruction 索引不可用：%s", error)
            if self.config.retrieval.hybrid_bge_alpha > 0 and not self._dense_loaded:
                try:
                    self._get_dense().load_index()
                    self._dense_loaded = True
                except FileNotFoundError as error:
                    logger.warning("dense 索引不可用：%s", error)
            if not (self._sparse_loaded or self._instruction_loaded or self._dense_loaded):
                raise RuntimeError("hybrid 没有可用后端")

    def _chunk_index(self) -> dict[tuple[str, int, int], int]:
        """建立 chunk 偏移到索引行号的映射，用于跨后端对齐分数。"""
        if self._chunk_index_cache is None:
            sparse = self._get_sparse()
            if (
                sparse.chunk_doc_ids is None
                or sparse.chunk_starts is None
                or sparse.chunk_ends is None
            ):
                self._chunk_index_cache = {}
            else:
                self._chunk_index_cache = {
                    (doc_id, int(start), int(end)): row
                    for row, (doc_id, start, end) in enumerate(
                        zip(
                            sparse.chunk_doc_ids,
                            sparse.chunk_starts,
                            sparse.chunk_ends,
                            strict=False,
                        )
                    )
                }
        return self._chunk_index_cache

    def _aligned(self, backend: DenseRetriever) -> bool:
        """稠密索引是否与稀疏索引共用同一份 chunk_map。"""
        sparse = self._get_sparse()
        return sparse.chunk_doc_ids is not None and len(sparse.chunk_doc_ids) == len(
            backend._doc_ids
        )

    def build_index(
        self, corpus_path: str | Path | None = None, method: RetrievalMethod | None = None
    ) -> None:
        """TODO：离线构建选定方法的索引；hybrid 构建三类索引并保存配置。"""
        target = method or self.config.retrieval.method
        path = corpus_path if corpus_path is not None else self.config.document.corpus_path
        if target in ("sparse", "hybrid"):
            self._get_sparse().build_index(path)
        if target in ("dense", "hybrid"):
            self._get_dense().build_index(path)
        if target == "instruction":
            self._get_instruction().build_index(path)

    def load_index(self, method: RetrievalMethod | None = None) -> None:
        """TODO：默认 config.retrieval.method；经共享 chroma 模块打开所需模型/集合。

        dense 使用 encoder，instruction 使用 instruction，hybrid 加载两者和 BM25。
        所有路径和模型来自 self.config；查询之间共用实例。
        """
        self._ensure_backends(method or self.config.retrieval.method)

    def _doc_level_hybrid(
        self, query: str, count: int, pool: int, instruction: str | None
    ) -> list[RetrievalResult]:
        """索引粒度不一致时的回退：文档级 BM25 + 稠密融合。"""
        sparse = self._get_sparse().retrieve(query, pool)
        if not sparse:
            return []
        reranker: DenseRetriever | None = None
        if self._instruction_loaded:
            reranker = self._get_instruction()
        elif self._dense_loaded:
            reranker = self._get_dense()
        if reranker is None:
            return sparse[:count]

        top = max(item.score for item in sparse) or 1.0
        dense_scores = reranker.score_documents(
            query, [item.doc_id for item in sparse], instruction
        )
        alpha = self.config.retrieval.hybrid_alpha
        by_doc = {item.doc_id: item for item in sparse}
        ordered = sorted(
            by_doc,
            key=lambda doc_id: (
                -(by_doc[doc_id].score / top + alpha * dense_scores.get(doc_id, 0.0))
            ),
        )
        return [
            by_doc[doc_id].model_copy(
                update={"score": by_doc[doc_id].score / top + alpha * dense_scores.get(doc_id, 0.0)}
            )
            for doc_id in ordered[:count]
        ]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        method: RetrievalMethod | None = None,
        instruction: str | None = None,
    ) -> list[RetrievalResult]:
        """输入搜索词、数量、方法和检索意图，输出统一排序结果。

        TODO：sparse/dense/instruction 路由对应后端；hybrid 多路召回再融合。
        instruction 仅用于指令检索分支。返回原始 doc_id 供 DocumentStore 读取。
        省略参数使用 self.config.retrieval 与 self.config.embedding 中的默认值。
        真正完成查询但无命中时才返回 []；未实现必须抛出 NotImplementedError。
        """
        target = method or self.config.retrieval.method
        count = top_k if top_k is not None else self.config.retrieval.top_k
        if count <= 0:
            return []

        self._ensure_backends(target)
        if target == "instruction":
            return self._get_instruction().retrieve(query, count, instruction)
        if target == "sparse":
            return self._get_sparse().retrieve(query, count)
        if target == "dense":
            return self._get_dense().retrieve(query, count)

        # hybrid：chunk 级分数融合（BM25 + bge + Qwen3），文档取最大。
        pool = max(self.config.retrieval.hybrid_pool, count)
        sparse = self._get_sparse()
        chunks = sparse.retrieve_chunks(query, pool)
        if not chunks:
            return []

        top = max(item[3] for item in chunks) or 1.0
        fused: np.ndarray = np.array([item[3] / top for item in chunks], dtype=np.float32)

        index = self._chunk_index()
        applied = False
        if index:
            rows = np.array(
                [index[(doc, start, end)] for doc, start, end, _ in chunks],
                dtype=np.int64,
            )
            if self._instruction_loaded and self._aligned(self._get_instruction()):
                instruction_backend = self._get_instruction()
                vector = instruction_backend._query_vector(query, instruction)
                fused = (
                    fused
                    + self.config.retrieval.hybrid_instruction_alpha
                    * instruction_backend._scores(vector.astype(np.float32))[rows]
                ).astype(np.float32)
                applied = True
            if self._dense_loaded and self._aligned(self._get_dense()):
                dense_backend = self._get_dense()
                vector = dense_backend._query_vector(query, instruction)
                fused = (
                    fused
                    + self.config.retrieval.hybrid_bge_alpha
                    * dense_backend._scores(vector.astype(np.float32))[rows]
                ).astype(np.float32)
                applied = True

        if not applied:
            return self._doc_level_hybrid(query, count, pool, instruction)

        results: list[RetrievalResult] = []
        for position in np.argsort(-fused):
            doc_id, start, end, _ = chunks[int(position)]
            if any(item.doc_id == doc_id for item in results):
                continue
            snippet, source_url = sparse._snippet(doc_id, start, end)
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    score=float(fused[int(position)]),
                    snippet=snippet,
                    source_url=source_url,
                )
            )
            if len(results) >= count:
                break
        return results
