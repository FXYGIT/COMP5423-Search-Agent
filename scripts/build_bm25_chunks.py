"""Build a chunk-level BM25S index from the corpus with original-text offsets.

The index stores one row per chunk (paragraph packing to ``--chunk-tokens``),
plus ``chunk_map.jsonl`` mapping every chunk back to its document and character
span, so results can be traced to the exact place in the original text.

Usage:
    python scripts/build_bm25_chunks.py --output indexes/bm25s-clean-1000
    python scripts/build_bm25_chunks.py --limit 2000 --output /tmp/test-index

Memory: tokenising ~580k chunks of 1000 tokens needs roughly 20 GB of RAM
(bm25s keeps the full tokenised corpus in memory while building). On a 32 GB
machine this fits; use ``--limit`` for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

TOKEN = re.compile(r"\w\w+")
PARAGRAPH = re.compile(r"(?s)\S(?:.*?)(?=\n[ \t]*\n|\Z)")
CHUNK_MAP_NAME = "chunk_map.jsonl"


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """按空行切段，返回每段在原文中的 (start, end)。"""
    return [(match.start(), match.end()) for match in PARAGRAPH.finditer(text)]


def _token_split(text: str, start: int, end: int, limit: int) -> Iterator[tuple[int, int]]:
    """把超长段落按 token 数切成不超过 limit 的片段，保留字符偏移。"""
    positions = [(m.start() + start, m.end() + start) for m in TOKEN.finditer(text, start, end)]
    if not positions:
        return
    for index in range(0, len(positions), limit):
        block = positions[index : index + limit]
        yield block[0][0], block[-1][1]


def chunk_spans(text: str, chunk_tokens: int) -> Iterator[tuple[int, int]]:
    """段落打包：相邻段落拼到 chunk_tokens 上限，超长段落再按 token 切。"""
    current_start = -1
    current_end = -1
    current_count = 0

    def flush() -> Iterator[tuple[int, int]]:
        nonlocal current_start, current_end, current_count
        if current_start >= 0:
            yield current_start, current_end
        current_start = current_end = -1
        current_count = 0

    for start, end in paragraph_spans(text):
        count = sum(1 for _ in TOKEN.finditer(text, start, end))
        if count == 0:
            continue
        if count > chunk_tokens:
            yield from flush()
            yield from _token_split(text, start, end, chunk_tokens)
            continue
        if current_count + count > chunk_tokens:
            yield from flush()
        if current_start < 0:
            current_start = start
        current_end = end
        current_count += count
    yield from flush()


def build(
    corpus_path: Path,
    output: Path,
    chunk_tokens: int,
    limit: int | None,
    prepend_title: bool = False,
) -> None:
    import bm25s
    import numpy as np
    from bm25s.tokenization import Tokenized, Tokenizer
    from datasets import load_from_disk

    from search_agent.core.document.cleaning import parse_fields

    dataset = load_from_disk(str(corpus_path))
    rows = len(dataset) if limit is None else min(limit, len(dataset))
    print(
        f"corpus={corpus_path} docs={rows:,} chunk_tokens={chunk_tokens} "
        f"prepend_title={prepend_title}",
        flush=True,
    )

    tokenizer = Tokenizer()
    token_ids: list[np.ndarray] = []
    chunk_rows: list[tuple[str, int, int]] = []
    started = time.perf_counter()

    for batch_start in range(0, rows, 256):
        batch_end = min(batch_start + 256, rows)
        batch = dataset.select(range(batch_start, batch_end))
        texts: list[str] = []
        for doc_id, text in zip(batch["docid"], batch["text"], strict=False):
            raw = str(text)
            title = parse_fields(raw)[0].get("title", "") if prepend_title else ""
            for start, end in chunk_spans(raw, chunk_tokens):
                chunk = raw[start:end]
                chunk_rows.append((str(doc_id), start, end))
                texts.append(f"{title}\n{chunk}" if title else chunk)
        ids = tokenizer.tokenize(texts, update_vocab=True, return_as="ids", show_progress=False)
        token_ids.extend(np.asarray(row, dtype=np.int32) for row in ids)
        if (batch_end % 2560) == 0:
            print(
                f"  docs {batch_end:,}/{rows:,} chunks {len(chunk_rows):,} "
                f"{time.perf_counter() - started:.0f}s",
                flush=True,
            )

    print(f"chunks={len(chunk_rows):,}, tokenising done, building index...", flush=True)
    tokenized = Tokenized(ids=token_ids, vocab=tokenizer.get_vocab_dict())
    retriever = bm25s.BM25(k1=1.5, b=0.75, method="lucene")
    retriever.index(tokenized, show_progress=False)

    output.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output), show_progress=False)
    with (output / CHUNK_MAP_NAME).open("w", encoding="utf-8") as handle:
        for doc_id, start, end in chunk_rows:
            handle.write(
                json.dumps({"doc_id": doc_id, "start": start, "end": end}, ensure_ascii=False)
                + "\n"
            )
    print(f"saved {output} ({len(chunk_rows):,} chunks)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(PROJECT / "data/browsecomp_plus_corpus"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prepend-title",
        action="store_true",
        help="tokenize 时在每段前加上文档标题（偏移仍指向原文）",
    )
    args = parser.parse_args()
    build(
        Path(args.corpus),
        Path(args.output),
        args.chunk_tokens,
        args.limit,
        prepend_title=args.prepend_title,
    )


if __name__ == "__main__":
    main()
