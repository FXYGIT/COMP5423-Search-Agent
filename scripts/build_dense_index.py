"""Build a dense/instruction index for the fixed corpus (numpy memmap backend)."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from search_agent.core.config import AppConfig  # noqa: E402
from search_agent.core.retrieval.dense import DenseRetriever  # noqa: E402
from search_agent.core.retrieval.instruction import InstructionRetriever  # noqa: E402

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake() -> None:
    """Windows 下阻止空闲睡眠；Linux 上无操作。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(  # type: ignore[attr-defined]
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
    except Exception as error:  # pragma: no cover
        print(f"keep_awake failed: {error}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--kind", choices=("dense", "instruction"), default="dense")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--index-root", default=str(PROJECT / "indexes" / "dense"))
    parser.add_argument("--corpus", default=str(PROJECT / "data/browsecomp_plus_corpus"))
    parser.add_argument("--chunk-map", default="", help="chunk_map.jsonl 路径；与 BM25 对齐")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--embed-title", action="store_true", help="把标题前置到每段")
    parser.add_argument("--no-clean", action="store_true", help="不做段落清洗")
    parser.add_argument("--clean-boilerplate", action="store_true", help="额外删除导航/页脚行")
    args = parser.parse_args()

    keep_awake()
    config = AppConfig()
    config.chroma.path = Path(args.index_root)
    config.chroma.dense_collection = args.collection
    config.chroma.instruction_collection = args.collection
    if args.kind == "instruction":
        config.embedding.instruction_model = args.model
    else:
        config.embedding.dense_model = args.model
    config.embedding.device = "cuda"
    config.embedding.batch_size = args.batch_size
    config.embedding.trust_remote_code = args.trust_remote_code
    config.embedding.chunk_map = args.chunk_map
    config.embedding.max_length = args.max_length
    config.embedding.embed_title = args.embed_title
    config.embedding.clean_passage = not args.no_clean
    config.embedding.clean_boilerplate = args.clean_boilerplate
    config.document.corpus_path = Path(args.corpus)

    retriever = (
        InstructionRetriever(config) if args.kind == "instruction" else DenseRetriever(config)
    )
    started = time.perf_counter()
    print(
        f"build start model={args.model} collection={args.collection} "
        f"chunk_map={args.chunk_map or '(self-split)'} max_length={args.max_length} "
        f"embed_title={args.embed_title} clean={not args.no_clean}",
        flush=True,
    )
    retriever.build_index(config.document.corpus_path)
    print(f"done in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
