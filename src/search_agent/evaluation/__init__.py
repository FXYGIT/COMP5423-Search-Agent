"""离线批量评估；不在导入时打开数据库、加载模型或读取参考答案。"""

from .metrics import evaluate, normalize_answer
from .runner import run_questions
from .schemas import EvaluationQuestion, EvaluationReport, PredictionRecord

__all__ = [
    "EvaluationQuestion",
    "EvaluationReport",
    "PredictionRecord",
    "evaluate",
    "normalize_answer",
    "run_questions",
]
