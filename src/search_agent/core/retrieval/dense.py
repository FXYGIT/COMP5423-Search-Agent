"""Encoder-based dense retrieval 骨架。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from search_agent.core.config import AppConfig, get_config
from search_agent.core.document.cleaning import clean_passage, parse_fields

from .schemas import RetrievalResult

PASSAGE_WORDS = 350
SNIPPET_CHARS = 500
CANDIDATE_PASSAGES = 2000
SEARCH_CHUNK = 50_000


class DenseRetriever:
    """使用 E5/BGE/GTE 等 encoder 编码查询和文档，进行向量检索。

    TODO：模型只初始化一次；明确 query/document 编码方式、归一化与相似度。
    和 instruction retriever 分开实现，避免把同一模型换提示词算作两类方法。
    """

    passage_words = PASSAGE_WORDS

    def __init__(self, config: AppConfig | None = None) -> None:
        """和建库共用模型、路径、collection，不下载模型或加载索引。"""
        self.config = config or get_config()
        self.index_path = self.config.chroma.path
        self.model_name = self.config.embedding.dense_model
        self.collection_name = self.config.chroma.dense_collection
        self._tokenizer: Any = None
        self._model: Any = None
        self._device = "cpu"
        self._embeddings: np.ndarray | None = None
        self._ram_embeddings: np.ndarray | None = None
        self._gpu_embeddings: Any = None
        self._doc_ids: list[str] = []
        self._starts: np.ndarray | None = None
        self._ends: np.ndarray | None = None
        self._dataset: Any = None
        self._doc_rows: dict[str, int] = {}
        self._doc_span: dict[str, tuple[int, int]] = {}

    # ---- 路径 ----

    def _dir(self) -> Path:
        """每个 collection 一个独立目录，便于多模型并存。"""
        return Path(self.index_path) / self.collection_name

    def _embedding_file(self) -> Path:
        return self._dir() / "embeddings.f16"

    # ---- 建库 ----

    def build_index(self, corpus_path: str | Path) -> None:
        """切块并批量编码固定语料，保存向量、原始 doc_id/偏移及模型配置。

        实现说明：使用 numpy memmap 顺序写向量，避免把全部向量放进内存；
        Chroma 的 HNSW 需要常驻 RAM，145 万 passage 会超出本机可用内存。
        配置了 ``EMBEDDING__CHUNK_MAP`` 时直接复用该文件的段落边界，
        从而与 BM25 的 chunk 完全对齐；否则按 ``passage_words`` 自行切分。
        """
        from datasets import load_from_disk  # type: ignore[import-untyped]

        dataset = load_from_disk(str(corpus_path))
        directory = self._dir()
        directory.mkdir(parents=True, exist_ok=True)
        self._ensure_model()
        dimension = int(self._model.config.hidden_size)

        embedding_path = self._embedding_file()
        meta_path = directory / "meta.jsonl"
        checkpoint_path = directory / "checkpoint.json"
        resume = self._resume_offset(checkpoint_path, embedding_path, meta_path, dimension)
        batch_texts: list[str] = []
        batch_meta: list[dict[str, Any]] = []
        count = 0
        skipped = 0
        seen = 0
        next_checkpoint = 20000
        with (
            embedding_path.open("ab" if resume else "wb") as embedding_file,
            meta_path.open("a" if resume else "w", encoding="utf-8") as meta_file,
        ):
            for doc_id, start, end, title, piece in self._iter_passages(dataset):
                seen += 1
                if skipped < resume:
                    skipped += 1
                    continue
                batch_texts.append(self._prepare_text(piece, title))
                batch_meta.append({"doc_id": doc_id, "start": start, "end": end})
                if len(batch_texts) >= self.config.embedding.batch_size:
                    count += self._write_batch(embedding_file, meta_file, batch_texts, batch_meta)
                    if count >= next_checkpoint:
                        embedding_file.flush()
                        meta_file.flush()
                        checkpoint_path.write_text(
                            json.dumps(
                                {
                                    "count": resume + count,
                                    "model": self.model_name,
                                    "dimension": dimension,
                                }
                            ),
                            encoding="utf-8",
                        )
                        next_checkpoint += 20000
                if seen % 50000 == 0:
                    print(
                        f"[dense] {seen:,} passages, {count:,} new (resumed at {resume:,})",
                        flush=True,
                    )
            count += self._write_batch(embedding_file, meta_file, batch_texts, batch_meta)

        info = {
            "model": self.model_name,
            "dimension": dimension,
            "count": resume + count,
            "passage_words": self.passage_words,
            "max_length": int(self.config.embedding.max_length),
            "embed_title": bool(self.config.embedding.embed_title),
            "clean_passage": bool(self.config.embedding.clean_passage),
            "chunk_map": self.config.embedding.chunk_map,
            "normalized": bool(self.config.embedding.normalize),
        }
        (directory / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        checkpoint_path.unlink(missing_ok=True)
        print(f"[dense] built {resume + count:,} passages -> {directory}", flush=True)

    def _iter_passages(self, dataset: Any) -> Iterator[tuple[str, int, int, str, str]]:
        """产出 (doc_id, start, end, title, 文本)：优先复用 chunk_map 的边界。"""
        chunk_map = self.config.embedding.chunk_map
        if chunk_map:
            yield from self._iter_chunk_map(dataset, Path(chunk_map))
            return
        for batch in dataset.iter(batch_size=256):
            for doc_id, text in zip(batch["docid"], batch["text"], strict=False):
                raw = str(text)
                title = parse_fields(raw)[0].get("title", "")
                for start, end, piece in _passages(raw, self.passage_words):
                    yield str(doc_id), start, end, title, piece

    def _iter_chunk_map(self, dataset: Any, path: Path) -> Iterator[tuple[str, int, int, str, str]]:
        """按 chunk_map.jsonl 的字符边界复用 BM25 的 chunk，保证两类索引同粒度。"""
        rows = {str(doc_id): row for row, doc_id in enumerate(dataset["docid"])}
        cached_id = ""
        cached_text = ""
        cached_title = ""
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                meta = json.loads(line)
                doc_id = str(meta["doc_id"])
                if doc_id != cached_id:
                    cached_id = doc_id
                    cached_text = str(dataset[rows[doc_id]]["text"])
                    cached_title = parse_fields(cached_text)[0].get("title", "")
                start, end = int(meta["start"]), int(meta["end"])
                yield doc_id, start, end, cached_title, cached_text[start:end]

    def _prepare_text(self, piece: str, title: str) -> str:
        """编码前的文本：可选清洗 + 可选标题前置。"""
        if self.config.embedding.clean_passage:
            text = clean_passage(piece, remove_boilerplate=self.config.embedding.clean_boilerplate)
        else:
            text = piece
        if self.config.embedding.embed_title and title:
            return f"{title}\n{text}"
        return text

    def _resume_offset(
        self,
        checkpoint_path: Path,
        embedding_path: Path,
        meta_path: Path,
        dimension: int,
    ) -> int:
        """读取断点并截断未完成的尾部；没有有效断点时从 0 开始。"""
        if not (checkpoint_path.exists() and embedding_path.exists() and meta_path.exists()):
            return 0
        try:
            state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0
        if state.get("model") != self.model_name:
            return 0
        if int(state.get("dimension", -1)) != dimension:
            return 0
        count = int(state.get("count", 0))
        if count <= 0:
            return 0
        with embedding_path.open("r+b") as handle:
            handle.truncate(count * dimension * 2)
        temporary = meta_path.with_suffix(".tmp")
        with (
            meta_path.open(encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8") as target,
        ):
            for index, line in enumerate(source):
                if index >= count:
                    break
                target.write(line)
        temporary.replace(meta_path)
        print(f"[dense] resuming from {count:,} passages", flush=True)
        return count

    def _write_batch(
        self,
        embedding_file: Any,
        meta_file: Any,
        texts: list[str],
        metas: list[dict[str, Any]],
    ) -> int:
        """编码一个批次并顺序写盘，返回写入条数。"""
        if not texts:
            return 0
        vectors = self._encode(texts)
        embedding_file.write(vectors.astype(np.float16).tobytes())
        for meta in metas:
            meta_file.write(json.dumps(meta) + "\n")
        written = len(texts)
        texts.clear()
        metas.clear()
        return written

    def _ensure_model(self) -> None:
        """模型与 tokenizer 只加载一次；设备和精度来自共享配置。"""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = self.config.embedding.device
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        trust = self.config.embedding.trust_remote_code
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust)
        self._model = AutoModel.from_pretrained(
            self.model_name, dtype=dtype, trust_remote_code=trust
        )
        if self._device != "cpu":
            self._model.to(self._device)
        self._model.eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        """CLS 池化 + L2 归一化（BGE/GTE 系列的标准池化方式）。"""
        import torch

        self._ensure_model()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self.config.embedding.max_length),
            return_tensors="pt",
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(self._device),
            "attention_mask": encoded["attention_mask"].to(self._device),
        }
        with torch.inference_mode():
            output = self._model(**inputs)
        pooled = output.last_hidden_state[:, 0]
        if self.config.embedding.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.float().cpu().numpy()  # type: ignore[no-any-return]

    # ---- 加载 ----

    def load_index(self) -> None:
        """TODO：加载向量库及 ID 映射，保证查询使用建库时的同一个模型。"""
        from datasets import load_from_disk

        directory = self._dir()
        info_path = directory / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"dense index not found: {directory}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        count = int(info["count"])
        dimension = int(info["dimension"])

        self._embeddings = np.memmap(
            self._embedding_file(), dtype=np.float16, mode="r", shape=(count, dimension)
        )
        doc_ids: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        with (directory / "meta.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                meta = json.loads(line)
                doc_ids.append(str(meta["doc_id"]))
                starts.append(int(meta["start"]))
                ends.append(int(meta["end"]))
        self._doc_ids = doc_ids
        self._starts = np.asarray(starts, dtype=np.int64)
        self._ends = np.asarray(ends, dtype=np.int64)
        self._ensure_model()

        self._dataset = load_from_disk(str(self.config.document.corpus_path))
        self._doc_rows = {str(doc_id): row for row, doc_id in enumerate(self._dataset["docid"])}

        # passage 按文档顺序连续存放，记录每个文档的区间以便重排时取最大值。
        spans: dict[str, tuple[int, int]] = {}
        first = 0
        for index, doc_id in enumerate(doc_ids):
            if index == 0:
                current = doc_id
                first = 0
            elif doc_id != current:
                spans[current] = (first, index)
                current = doc_id
                first = index
        if doc_ids:
            spans[current] = (first, len(doc_ids))
        self._doc_span = spans

    # ---- 查询 ----

    def _query_vector(self, query: str, instruction: str | None = None) -> np.ndarray:
        """查询编码入口；指令检索会在这里加前缀。"""
        return self._encode([query])[0]  # type: ignore[no-any-return]

    def score_documents(
        self, query: str, doc_ids: list[str], instruction: str | None = None
    ) -> dict[str, float]:
        """给指定文档打分（取该文档所有 passage 的最高相似度），供融合使用。"""
        if self._embeddings is None:
            raise RuntimeError("Load the dense index before scoring.")
        vector = self._query_vector(query, instruction).astype(np.float32)
        scores = self._scores(vector)
        result: dict[str, float] = {}
        for doc_id in doc_ids:
            span = self._doc_span.get(doc_id)
            if span is None:
                continue
            lo, hi = span
            if hi > lo:
                result[doc_id] = float(scores[lo:hi].max())
        return result

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        instruction: str | None = None,
    ) -> list[RetrievalResult]:
        """TODO：复用 chroma.create_embeddings/open_chroma；数量默认 config.retrieval.top_k。

        编码查询、向量召回、块级结果按文档归并，返回 top_k 篇原文。
        """
        limit = top_k if top_k is not None else self.config.retrieval.top_k
        if limit <= 0:
            return []
        if self._embeddings is None:
            raise RuntimeError("Load the dense index before retrieving.")

        vector = self._query_vector(query, instruction).astype(np.float32)
        scores = self._scores(vector)
        starts = self._starts
        ends = self._ends
        if starts is None or ends is None:
            raise RuntimeError("Load the dense index before retrieving.")
        candidate = min(CANDIDATE_PASSAGES, scores.shape[0])
        part = np.argpartition(-scores, candidate - 1)[:candidate]
        order = part[np.argsort(-scores[part])]

        results: list[RetrievalResult] = []
        seen: set[str] = set()
        for index in order:
            doc_id = self._doc_ids[int(index)]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            snippet, source_url = self._snippet(doc_id, int(starts[index]), int(ends[index]))
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    score=float(scores[index]),
                    snippet=snippet,
                    source_url=source_url,
                )
            )
            if len(results) >= limit:
                break
        return results

    def _scores(self, vector: np.ndarray) -> np.ndarray:
        """计算查询与全部 passage 的余弦相似度。

        有 GPU 时把向量一次性分块搬到显存（约 2.7 GB），之后每条查询只是一次
        矩阵乘；否则分块读 memmap 计算，控制峰值内存。
        """
        assert self._embeddings is not None
        if self._device.startswith("cuda"):
            import torch

            if self._gpu_embeddings is None:
                total, dimension = self._embeddings.shape
                tensor = torch.empty((total, dimension), dtype=torch.float16, device=self._device)
                for start in range(0, total, SEARCH_CHUNK):
                    end = min(start + SEARCH_CHUNK, total)
                    block = np.array(self._embeddings[start:end], copy=True)
                    tensor[start:end] = torch.from_numpy(block).to(self._device)
                self._gpu_embeddings = tensor
            query = torch.from_numpy(vector).to(self._device).to(torch.float16)
            return (self._gpu_embeddings @ query).float().cpu().numpy()  # type: ignore[no-any-return]

        total = self._embeddings.shape[0]
        if self._ram_embeddings is None:
            # CPU 查询：把整份向量读进内存一次，避免每条 query 重复读盘。
            self._ram_embeddings = np.asarray(self._embeddings, dtype=np.float32)
        scores = np.empty(total, dtype=np.float32)
        for start in range(0, total, SEARCH_CHUNK):
            end = min(start + SEARCH_CHUNK, total)
            scores[start:end] = self._ram_embeddings[start:end] @ vector
        return scores

    def _snippet(self, doc_id: str, start: int, end: int) -> tuple[str, str | None]:
        """按原文偏移取片段和来源 URL。"""
        row = self._doc_rows.get(doc_id)
        if row is None or self._dataset is None:
            return "", None
        record = self._dataset[row]
        text = str(record["text"])[start:end][:SNIPPET_CHARS]
        url = record.get("url")
        return text, str(url) if url is not None else None


def _passages(text: str, words: int = PASSAGE_WORDS) -> list[tuple[int, int, str]]:
    """按段落切分并把相邻段落打包到约 ``words`` 词，保留原文字符偏移。"""
    import re

    paragraph = re.compile(r"(?:\r?\n)[ \t\r]*(?:\r?\n)+")
    sentence = re.compile(r"(?<=[.!?])\s+")

    spans: list[tuple[int, int]] = []
    start = 0
    for separator in paragraph.finditer(text):
        spans.extend(_trim_split(text, start, separator.start(), sentence, words))
        start = separator.end()
    spans.extend(_trim_split(text, start, len(text), sentence, words))

    passages: list[tuple[int, int, str]] = []
    current_start: int | None = None
    current_end: int | None = None
    current_words = 0
    for seg_start, seg_end in spans:
        count = len(text[seg_start:seg_end].split())
        if count == 0:
            continue
        if current_start is None:
            current_start, current_end, current_words = seg_start, seg_end, count
        elif current_words + count <= words:
            current_end, current_words = seg_end, current_words + count
        else:
            assert current_end is not None
            passages.append((current_start, current_end, text[current_start:current_end]))
            current_start, current_end, current_words = seg_start, seg_end, count
    if current_start is not None and current_end is not None:
        passages.append((current_start, current_end, text[current_start:current_end]))
    return passages


def _trim_split(
    text: str, start: int, end: int, sentence: Any, words: int = PASSAGE_WORDS
) -> list[tuple[int, int]]:
    """去首尾空白；超过 ``words`` 时按句子边界切开。"""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return []
    if len(text[start:end].split()) <= words:
        return [(start, end)]
    parts: list[tuple[int, int]] = []
    cursor = start
    for match in sentence.finditer(text, start, end):
        if match.start() > cursor:
            parts.append((cursor, match.start()))
        cursor = match.end()
    if cursor < end:
        parts.append((cursor, end))
    if not parts:
        return [(start, end)]
    # 先算一次每句词数，避免在循环里反复 split 造成 O(n^2)。
    counts = [len(text[seg_start:seg_end].split()) for seg_start, seg_end in parts]
    result: list[tuple[int, int]] = []
    current_start, current_end = parts[0]
    current_words = counts[0]
    for (next_start, next_end), next_words in zip(parts[1:], counts[1:], strict=False):
        if current_words + next_words <= words:
            current_end = next_end
            current_words += next_words
        else:
            result.append((current_start, current_end))
            current_start, current_end, current_words = next_start, next_end, next_words
    result.append((current_start, current_end))
    return result
