"""作业 normalized EM 与用量指标；参考答案仅在这里参与计算。"""

import re
import string

from .schemas import EvaluationQuestion, EvaluationReport, PredictionRecord, QuestionResult


def normalize_answer(answer: str) -> str:
    """按题目定义依次小写、去英文标点、去 a/an/the、合并空白。"""
    text = answer.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def evaluate(
    questions: list[EvaluationQuestion], records: list[PredictionRecord]
) -> EvaluationReport:
    """按原始 ID 对齐；要求每题恰好一条预测，输出准确率和实际用量汇总。

    EM 比较整个 final answer，不通过参考答案提取或修补模型输出。
    无标签测试集 EM=None；不把用量指标换算成未知的班级百分位分数。
    """
    expected = {question.id for question in questions}
    indexed = {record.id: record for record in records}
    if len(expected) != len(questions) or len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("预测 ID 必须唯一且完整对应数据集 ID，不可缺题或多题。")
    if not questions:
        raise ValueError("评估数据集不能为空。")
    results = []
    for question in questions:
        record = indexed[question.id]
        if record.question != question.question:
            raise ValueError(f"ID {question.id} 的问题文本与数据集不一致。")
        match = (
            None
            if question.answer is None
            else (normalize_answer(record.answer) == normalize_answer(question.answer))
        )
        results.append(
            QuestionResult(
                id=question.id,
                prediction=record.answer,
                reference=question.answer,
                exact_match=match,
                tool_calls=record.tool_calls,
                total_tokens=record.total_tokens,
            )
        )
    labelled = [row for row in results if row.exact_match is not None]
    total_calls = sum(row.tool_calls for row in results)
    total_tokens = sum(row.total_tokens for row in results)
    return EvaluationReport(
        question_count=len(questions),
        labelled_count=len(labelled),
        normalized_em=sum(row.exact_match is True for row in labelled) / len(labelled)
        if labelled
        else None,
        total_tool_calls=total_calls,
        total_tokens=total_tokens,
        mean_tool_calls=total_calls / len(questions),
        mean_tokens=total_tokens / len(questions),
        results=results,
    )
