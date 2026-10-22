"""评估接线使用独立假 Agent，不调用模型 API 或下载 embedding。"""

import json
import os
import subprocess
import sys

import pytest
from langchain_core.messages import AIMessage


def test_normalized_em_and_total_usage():
    from search_agent.evaluation.metrics import evaluate
    from search_agent.evaluation.schemas import EvaluationQuestion, PredictionRecord

    questions = [
        EvaluationQuestion(id="a", question="Capital?", answer="The Paris"),
        EvaluationQuestion(id="b", question="Name?", answer="Alice"),
    ]
    records = [
        PredictionRecord(
            id="b", question="Name?", answer="Bob", trajectory=[{}], tool_calls=3, total_tokens=80
        ),
        PredictionRecord(
            id="a",
            question="Capital?",
            answer=" PARIS! ",
            trajectory=[{}],
            tool_calls=1,
            total_tokens=20,
        ),
    ]
    report = evaluate(questions, records)
    assert report.normalized_em == 0.5
    assert report.total_tool_calls == 4
    assert report.total_tokens == 100
    assert report.mean_tool_calls == 2
    assert report.mean_tokens == 50
    assert [row.id for row in report.results] == ["a", "b"]


def test_missing_predictions_are_not_silently_excluded():
    from search_agent.evaluation.metrics import evaluate
    from search_agent.evaluation.schemas import EvaluationQuestion

    with pytest.raises(ValueError, match="ID"):
        evaluate([EvaluationQuestion(id="a", question="Question")], [])


def test_unlabelled_test_set_has_no_accuracy():
    from search_agent.evaluation.metrics import evaluate
    from search_agent.evaluation.schemas import EvaluationQuestion, PredictionRecord

    report = evaluate(
        [EvaluationQuestion(id="a", question="Q")],
        [
            PredictionRecord(
                id="a", question="Q", answer="A", trajectory=[{}], tool_calls=0, total_tokens=5
            )
        ],
    )
    assert report.normalized_em is None
    assert report.results[0].exact_match is None


def test_batch_isolates_questions_and_writes_six_fields(tmp_path):
    from search_agent.evaluation.runner import run_questions, write_predictions
    from search_agent.evaluation.schemas import EvaluationQuestion

    class Agent:
        def invoke(self, state):
            # A previous answer, reference label, or TODO must never leak into the next question.
            assert set(state) == {"messages"}
            assert len(state["messages"]) == 1
            assert "secret" not in state["messages"][0].content
            return {
                "messages": [*state["messages"], AIMessage(content="Paris")],
                "trajectory": [{"type": "model", "text": "Paris"}],
                "tool_calls": 1,
                "total_tokens": 10,
            }

    questions = [EvaluationQuestion(id=str(i), question=f"Q{i}", answer="secret") for i in range(2)]
    records = run_questions(Agent(), questions)
    output = tmp_path / "test_prediction.jsonl"
    write_predictions(records, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["0", "1"]
    assert set(rows[0]) == {"id", "question", "answer", "trajectory", "tool_calls", "total_tokens"}
    assert rows[1]["total_tokens"] == 10


def test_unimplemented_statistics_do_not_become_zero():
    from search_agent.evaluation.runner import prediction_from_state
    from search_agent.evaluation.schemas import EvaluationQuestion

    with pytest.raises(NotImplementedError, match="TrajectoryMiddleware"):
        prediction_from_state(
            EvaluationQuestion(id="a", question="Q"), {"messages": [AIMessage(content="A")]}
        )


def test_local_dataset_and_reports(tmp_path):
    from datasets import Dataset, DatasetDict

    from search_agent.evaluation.dataset import load_questions
    from search_agent.evaluation.metrics import evaluate
    from search_agent.evaluation.report import write_report
    from search_agent.evaluation.runner import write_predictions
    from search_agent.evaluation.schemas import PredictionRecord

    dataset_path = tmp_path / "val"
    DatasetDict(
        {"validation": Dataset.from_dict({"id": ["a"], "question": ["Q"], "answer": ["The A"]})}
    ).save_to_disk(dataset_path)
    questions = load_questions(dataset_path, split="validation")
    report = evaluate(
        questions,
        [
            PredictionRecord(
                id="a", question="Q", answer="a", trajectory=[{}], tool_calls=2, total_tokens=30
            )
        ],
    )
    write_report(report, tmp_path)
    assert (
        json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))[
            "normalized_em"
        ]
        == 1.0
    )
    assert "100.00%" in (tmp_path / "evaluation_report.md").read_text(encoding="utf-8")

    # 检查独立复评脚本实际读取 JSONL 并写出同样结果，不初始化模型或 Chroma。
    predictions = tmp_path / "test_prediction.jsonl"
    write_predictions(
        [
            PredictionRecord(
                id="a",
                question="Q",
                answer="a",
                trajectory=[{"type": "test_fixture"}],
                tool_calls=2,
                total_tokens=30,
            )
        ],
        predictions,
    )
    output_dir = tmp_path / "cli_report"
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_predictions.py"],
        env={
            **os.environ,
            "EVALUATION__DATASET_PATH": str(dataset_path),
            "EVALUATION__PREDICTIONS_PATH": str(predictions),
            "EVALUATION__OUTPUT_DIR": str(output_dir),
            "EVALUATION__LIMIT": "",
            "EVALUATION__SPLIT": "validation",
            "LOGGING__PATH": str(tmp_path / "evaluation.log"),
        },
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == b"" and completed.stderr == b""
    assert (
        json.loads((output_dir / "evaluation_report.json").read_text(encoding="utf-8"))[
            "total_tokens"
        ]
        == 30
    )
