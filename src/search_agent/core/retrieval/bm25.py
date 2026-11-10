"""稀疏检索：加载 BM25S 索引，按原始 doc_id 返回文档、片段和来源 URL。

查询打分在 BM25S 预计算的 impact 上做 idf 幂次加权：

    score(d) = Σ_t idf(t) ** idf_power · s(t, d)

idf_power=0 等价于原始 BM25S 打分；idf_power=1 相当于每个词项再乘一次
idf（验证集上 R@10 约提升 2.5 个百分点）。idf 只来自语料统计量
（倒排区间长度），不使用任何标签。

索引目录里若存在 chunk_map.jsonl，则索引的一行是一个 chunk，检索结果会
归并回原始文档并用 chunk 的 start/end 切出 snippet；否则一行就是一篇文档。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bm25s  # type: ignore[import-untyped]
import numpy as np
from datasets import load_from_disk  # type: ignore[import-untyped]

from search_agent.core.config import AppConfig, get_config

from .schemas import RetrievalResult

CHUNK_MAP_NAME = "chunk_map.jsonl"
PARAMS_NAME = "params.index.json"
SNIPPET_CHARS = 500


class BM25Retriever:
    """维护 BM25 索引、原文 chunk 映射和 idf 权重，供统一 Retriever 调用。

    构造阶段不读语料、不加载索引；load_index 之后同一实例可复用查询。
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """读取索引目录与查询加权参数；构造阶段不加载任何数据。"""
        self.config = config or get_config()
        self.index_path: Path = self.config.retrieval.sparse_index_path
        self.idf_power: float = self.config.retrieval.idf_power
        self.candidate_chunks: int = self.config.retrieval.candidate_chunks
        self.retriever: bm25s.BM25 | None = None
        self.dataset: Any = None
        self.idf: np.ndarray | None = None
        self.num_docs = 0
        self.chunk_doc_ids: list[str] | None = None
        self.chunk_starts: np.ndarray | None = None
        self.chunk_ends: np.ndarray | None = None
        self.doc_rows: dict[str, int] = {}

    def build_index(self, corpus_path: str | Path, limit: int | None = 1000) -> None:
        """从固定语料建立 BM25S 索引并写盘；不在索引里重复保存原文。"""
        dataset = load_from_disk(str(corpus_path))
        rows = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))

        texts = rows["text"]
        corpus_tokens = bm25s.tokenize(texts, stopwords=None, show_progress=False)

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)

        self.index_path.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self.index_path), show_progress=False)

    def load_index(self) -> None:
        """从 index_path 恢复索引、idf 权重与 chunk 映射；后续查询复用。"""
        params_path = self.index_path / PARAMS_NAME
        if not params_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {self.index_path}")

        params = json.loads(params_path.read_text(encoding="utf-8"))
        self.num_docs = int(params["num_docs"])
        self.retriever = bm25s.BM25.load(str(self.index_path), mmap=True)
        indptr = np.asarray(self.retriever.scores["indptr"])
        doc_freq = np.diff(indptr).astype(np.float64)
        self.idf = np.log(1 + (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5)).astype(
            np.float32
        )

        self.dataset = load_from_disk(str(self.config.document.corpus_path))
        self.doc_rows = {str(doc_id): row for row, doc_id in enumerate(self.dataset["docid"])}

        chunk_map = self.index_path / CHUNK_MAP_NAME
        self.chunk_doc_ids = None
        self.chunk_starts = None
        self.chunk_ends = None
        if chunk_map.exists():
            self._load_chunk_map(chunk_map)

    def _load_chunk_map(self, path: Path) -> None:
        """读取 chunk 到原文的映射，保留原文字符偏移。"""
        doc_ids: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                meta = json.loads(line)
                doc_ids.append(str(meta["doc_id"]))
                starts.append(int(meta["start"]))
                ends.append(int(meta["end"]))
        self.chunk_doc_ids = doc_ids
        self.chunk_starts = np.asarray(starts, dtype=np.int64)
        self.chunk_ends = np.asarray(ends, dtype=np.int64)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """查询并返回按分数降序的文档结果，数量默认 config.retrieval.top_k。"""
        if self.retriever is None or self.idf is None:
            raise RuntimeError("Load the BM25 index before retrieving.")

        limit = top_k if top_k is not None else self.config.retrieval.top_k
        if limit <= 0:
            return []

        tokens = bm25s.tokenize([query], stopwords=None, show_progress=False, return_ids=False)[0]
        vocab_dict = self.retriever.vocab_dict
        ids = np.fromiter(
            (vocab_dict[token] for token in tokens if token in vocab_dict),
            dtype=np.int32,
        )
        if ids.size == 0:
            return []

        scores = self._scores(ids)
        candidate = min(self.candidate_chunks, self.num_docs)
        part = np.argpartition(-scores, candidate - 1)[:candidate]
        order = part[np.argsort(-scores[part])]

        results: list[RetrievalResult] = []
        seen: set[str] = set()
        for row in order:
            doc_id, start, end = self._row_location(int(row))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            snippet, source_url = self._snippet(doc_id, start, end)
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    score=float(scores[row]),
                    snippet=snippet,
                    source_url=source_url,
                )
            )
            if len(results) >= limit:
                break
        return results

    def retrieve_chunks(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[str, int, int, float]]:
        """返回 chunk 级结果 (doc_id, start, end, score)，供段落级融合使用。"""
        if self.retriever is None or self.idf is None:
            raise RuntimeError("Load the BM25 index before retrieving.")
        limit = top_k if top_k is not None else self.config.retrieval.top_k
        if limit <= 0:
            return []

        tokens = bm25s.tokenize([query], stopwords=None, show_progress=False, return_ids=False)[0]
        vocab_dict = self.retriever.vocab_dict
        ids = np.fromiter(
            (vocab_dict[token] for token in tokens if token in vocab_dict),
            dtype=np.int32,
        )
        if ids.size == 0:
            return []

        scores = self._scores(ids)
        candidate = min(limit, self.num_docs)
        part = np.argpartition(-scores, candidate - 1)[:candidate]
        order = part[np.argsort(-scores[part])]

        results: list[tuple[str, int, int, float]] = []
        for row in order:
            doc_id, start, end = self._row_location(int(row))
            results.append((doc_id, start, end, float(scores[row])))
        return results

    def _scores(self, ids: np.ndarray) -> np.ndarray:
        """把查询词项的预计算 impact 按 idf^power 加权后累加到每个 chunk 上。"""
        assert self.retriever is not None and self.idf is not None
        scores = self.retriever.scores
        indptr = scores["indptr"]
        indices = scores["indices"]
        data = scores["data"]

        starts = indptr[ids]
        ends = indptr[ids + 1]
        lengths = (ends - starts).astype(np.int64)
        doc_index = np.concatenate(
            [indices[start:end] for start, end in zip(starts, ends, strict=False)]
        )
        impacts = np.concatenate(
            [data[start:end] for start, end in zip(starts, ends, strict=False)]
        )
        if self.idf_power:
            weights = np.repeat(self.idf[ids] ** self.idf_power, lengths)
            impacts = impacts * weights
        return np.bincount(doc_index, weights=impacts, minlength=self.num_docs)

    def _row_location(self, row: int) -> tuple[str, int, int]:
        """把索引行号翻译成 (doc_id, 原文起止)；无 chunk 映射时整篇即一行。"""
        if self.chunk_doc_ids is not None:
            assert self.chunk_starts is not None and self.chunk_ends is not None
            return (
                self.chunk_doc_ids[row],
                int(self.chunk_starts[row]),
                int(self.chunk_ends[row]),
            )
        record = self.dataset[row]
        return str(record["docid"]), 0, len(record["text"])

    def _snippet(self, doc_id: str, start: int, end: int) -> tuple[str, str | None]:
        """按原文偏移切出片段并取来源 URL。"""
        row = self.doc_rows.get(doc_id)
        if row is None:
            return "", None
        record = self.dataset[row]
        text = str(record["text"])[start:end][:SNIPPET_CHARS]
        url = record.get("url")
        return text, str(url) if url is not None else None
