"""读取下载脚本保存的 Hugging Face Dataset / DatasetDict，不触发网络下载。"""

from pathlib import Path

from datasets import DatasetDict, load_from_disk  # type: ignore[import-untyped]

from .schemas import EvaluationQuestion


def load_questions(
    path: Path, *, split: str | None = None, limit: int | None = None
) -> list[EvaluationQuestion]:
    """输入本地数据目录，输出 id/question/可选 answer。

    项目验证集只有 validation split；测试集没有 answer 列。
    多 split 时显式指定，不自行选择。limit 仅用于联调，不按答案筛题。
    """
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split is None:
            if len(dataset) != 1:
                raise ValueError("数据集有多个 split，请使用 --split 指定。")
            split = next(iter(dataset))
        dataset = dataset[split]
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [
        EvaluationQuestion(id=str(row["id"]), question=row["question"], answer=row.get("answer"))
        for row in dataset
    ]
