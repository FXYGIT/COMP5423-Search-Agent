"""Emit chunk_map.jsonl only (no tokenising / no index build).

Uses the exact same chunking function as build_bm25_chunks.py, so dense indexes
built with this map are perfectly aligned with the BM25 chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))

from build_bm25_chunks import chunk_spans  # noqa: E402


def main() -> None:
    from datasets import load_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(PROJECT / "data/browsecomp_plus_corpus"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-tokens", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dataset = load_from_disk(args.corpus)
    rows = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for start_row in range(0, rows, 512):
            batch = dataset.select(range(start_row, min(start_row + 512, rows)))
            for doc_id, text in zip(batch["docid"], batch["text"], strict=False):
                for start, end in chunk_spans(str(text), args.chunk_tokens):
                    handle.write(
                        json.dumps(
                            {"doc_id": str(doc_id), "start": start, "end": end},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    count += 1
            if (start_row + 512) % 20480 == 0:
                print(f"docs {start_row + 512:,}/{rows:,} chunks {count:,}", flush=True)
    print(f"wrote {count:,} chunks -> {target}", flush=True)


if __name__ == "__main__":
    main()
