"""将同一份指标输出为机器可读 JSON 和便于查看的 Markdown。"""

from pathlib import Path

from .schemas import EvaluationReport


def write_report(report: EvaluationReport, output_dir: Path) -> None:
    """输出 evaluation_report.json/md；逐题参考答案仅出现在报告中。

    TODO：需要实验对比时记录运行配置/版本，不在本骨架生成排名或虚构分析结论。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    em = "无参考答案，不计算" if report.normalized_em is None else f"{report.normalized_em:.2%}"
    lines = [
        "# 本地评估报告",
        "",
        f"题目数：{report.question_count}；有标签题数：{report.labelled_count}。",
        "本报告只对应本次输入数据，不代表完整作业评分或最终测试集提交。",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| Normalized EM | {em} |",
        f"| 工具调用总数 | {report.total_tool_calls} |",
        f"| 每题平均工具调用 | {report.mean_tool_calls:.2f} |",
        f"| API token 总数 | {report.total_tokens} |",
        f"| 每题平均 API tokens | {report.mean_tokens:.2f} |",
        "",
        "## 逐题结果",
        "",
        "| ID | 匹配 | 预测 | 参考 | 调用 | Tokens |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.results:
        match = "—" if row.exact_match is None else ("正确" if row.exact_match else "错误")
        fields = [
            row.id,
            match,
            row.prediction,
            row.reference or "—",
            str(row.tool_calls),
            str(row.total_tokens),
        ]
        lines.append(
            "| "
            + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in fields)
            + " |"
        )
    (output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
