"""下载固定原文到统一 DOCUMENT__CORPUS_PATH，仅显式运行脚本时下载。"""

from datasets import load_dataset

from search_agent.core.config import get_config


def main() -> None:
    config = get_config()
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    corpus.save_to_disk(str(config.document.corpus_path))


if __name__ == "__main__":
    main()
