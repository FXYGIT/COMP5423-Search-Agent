"""从 .env 读取评估路径，对已有预测文件复评；不调用模型 API。"""

import argparse
import logging

from search_agent.core.config import get_config
from search_agent.core.logging_setup import setup_logger
from search_agent.evaluation.dataset import load_questions
from search_agent.evaluation.metrics import evaluate
from search_agent.evaluation.report import write_report
from search_agent.evaluation.runner import read_predictions


def main() -> None:
    """EVALUATION 配置决定数据、预测、split、limit 和报告位置。"""
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = get_config()
    setup_logger(config.logging)
    options = config.evaluation
    questions = load_questions(options.dataset_path, split=options.split, limit=options.limit)
    report = evaluate(questions, read_predictions(options.predictions_path))
    write_report(report, options.output_dir)
    logging.getLogger(__name__).info(
        "复评完成，题数=%d，EM=%s，工具调用=%d，tokens=%d",
        report.question_count,
        report.normalized_em,
        report.total_tool_calls,
        report.total_tokens,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("复评未完成")
        raise SystemExit(1) from None
