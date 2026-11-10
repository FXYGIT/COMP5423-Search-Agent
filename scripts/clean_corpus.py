"""Build the cleaned corpus dataset (docid, url, title, author, date, text)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from search_agent.core.document.cleaning import clean_document  # noqa: E402


def main() -> None:
    from datasets import Dataset, load_from_disk

    parser = argparse.ArgumentParser()
    parser.add_argument("--aggressive", action="store_true", help="同时过滤导航/页脚行")
    parser.add_argument("--name", default="browsecomp_plus_clean")
    args = parser.parse_args()

    source = load_from_disk(str(PROJECT / "data/browsecomp_plus_corpus"))
    stats = {"changed": 0, "titles": 0, "authors": 0}

    def rows() -> Iterator[dict]:
        for batch in source.iter(batch_size=1000):
            for docid, url, text in zip(batch["docid"], batch["url"], batch["text"], strict=False):
                cleaned = clean_document(str(text), remove_boilerplate=args.aggressive)
                stats["changed"] += 1 if cleaned.changed else 0
                stats["titles"] += 1 if cleaned.title else 0
                stats["authors"] += 1 if cleaned.author else 0
                yield {
                    "docid": str(docid),
                    "url": str(url),
                    "title": cleaned.title,
                    "author": cleaned.author,
                    "date": cleaned.date,
                    "text": cleaned.text,
                }

    dataset = Dataset.from_generator(rows, writer_batch_size=1000)
    target = PROJECT / "data" / args.name
    dataset.save_to_disk(str(target))
    print(
        f"docs={len(dataset):,} changed={stats['changed']:,} "
        f"titles={stats['titles']:,} authors={stats['authors']:,} aggressive={args.aggressive}"
    )
    print(f"saved {target}")


if __name__ == "__main__":
    main()
