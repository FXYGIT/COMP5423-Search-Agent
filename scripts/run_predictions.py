"""从 .env 读取全部配置，生成六字段预测 JSONL 和 JSON/Markdown 报告。"""

import argparse
import logging

from search_agent.core.config import get_config
from search_agent.core.logging_setup import setup_logger
from search_agent.evaluation.dataset import load_questions
from search_agent.evaluation.metrics import evaluate
from search_agent.evaluation.report import write_report
from search_agent.evaluation.runner import create_evaluation_agent, run_questions, write_predictions


def main() -> None:
    """共享配置→读取题目→逐题运行→评分与导出；业务 TODO 缺失时不生成假数据。"""
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = get_config()
    setup_logger(config.logging)
    options = config.evaluation
    questions = load_questions(options.dataset_path, split=options.split, limit=options.limit)
    agent = create_evaluation_agent(config)
    records = run_questions(agent, questions)
    report = evaluate(questions, records)
    write_predictions(records, options.predictions_path)
    write_report(report, options.output_dir)
    logging.getLogger(__name__).info(
        "评估完成，题数=%d，预测文件=%s", report.question_count, options.predictions_path
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("评估未完成")
        raise SystemExit(1) from None
