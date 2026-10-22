"""下载验证集到统一 EVALUATION__DATASET_PATH，仅显式运行脚本时下载。"""

from datasets import load_dataset

from search_agent.core.config import get_config


def main() -> None:
    config = get_config()
    dataset = load_dataset("HongruCai/COMP5423-BrowseComp-Plus-Val")
    dataset.save_to_disk(str(config.evaluation.dataset_path))


if __name__ == "__main__":
    main()
