"""批量执行与六字段预测导出，不在评估层重复实现 Agent 循环。"""

import logging
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from search_agent.core import create_search_agent
from search_agent.core.config import AppConfig, get_config
from search_agent.core.logging_setup import setup_logger
from search_agent.core.retrieval import Retriever
from search_agent.core.state import SearchState

from .chroma import create_evaluation_tools
from .schemas import EvaluationQuestion, PredictionRecord

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = """Answer the question using the fixed BrowseComp-Plus corpus.
Use the single retrieve tool and supporting document/skill tools as needed.
Search iteratively, collect evidence and verify the answer. Never use reference labels.
Your FINAL response must contain only the short answer, without explanations or citations.
Do not claim evidence from tools returning not_implemented.
"""


class EvaluationAgent(Protocol):
    """评估只依赖原生 invoke 接口，测试可注入离线 Agent。"""

    def invoke(self, input: SearchState) -> dict[str, Any]: ...


def create_evaluation_agent(config: AppConfig | None = None) -> EvaluationAgent:
    """建立 Chroma → 工具 → 共用 Agent 工厂的明确调用链。

    模型 API 配置沿用 .env。工厂初始化一次；每题全新 state，无 checkpointer。
    TODO 接口未补齐时明确中止，不退回未连接 Chroma 的默认占位工具。
    """
    config = config or get_config()
    setup_logger(config.logging)
    logger.info("初始化评估 Agent，检索方法=%s", config.retrieval.method)
    retriever = Retriever(config)
    retriever.load_index()
    tools = create_evaluation_tools(retriever, config)
    return create_search_agent(tools=tools, system_prompt=EVALUATION_PROMPT, config=config)


def prediction_from_state(question: EvaluationQuestion, state: dict[str, Any]) -> PredictionRecord:
    """读取 Agent 最终状态，将真实结果组装为提交记录。

    统计生产者是 TrajectoryMiddleware；评估不猜 token，也不拼造缺失轨迹。
    middleware 提供示例格式的 role 消息，PredictionRecord 校验逐条 usage 与总数。
    TODO：辅助模型/摘要实现后，也必须通过共用轨迹协议记录。
    """
    required = {"trajectory", "tool_calls", "total_tokens"}
    if not required.issubset(state):
        raise NotImplementedError("TrajectoryMiddleware 尚未提供完整轨迹和实际 API 用量。")
    if state["total_tokens"] is None:
        raise ValueError("模型响应缺少 API usage，不能导出完整 total_tokens；请检查流式用量支持。")
    final = state["messages"][-1]
    if not isinstance(final, AIMessage) or final.tool_calls:
        raise ValueError(f"题目 {question.id} 尚未产生最终答案。")
    return PredictionRecord(
        id=question.id,
        question=question.question,
        answer=final.text.strip(),
        trajectory=state["trajectory"],
        tool_calls=state["tool_calls"],
        total_tokens=state["total_tokens"],
    )


def run_questions(
    agent: EvaluationAgent, questions: list[EvaluationQuestion]
) -> list[PredictionRecord]:
    """逐题运行并返回预测列表。只传 question，不传 id/answer 或其他题目的状态。

    当前顺序执行，默认输入合法，不实现并发、重试和断点续跑。
    任一题执行失败时向上抛出，不将失败题悄悄排除在评估之外。
    """
    records = []
    for question in questions:
        logger.info("开始评估题目 id=%s", question.id)
        state = agent.invoke({"messages": [HumanMessage(content=question.question)]})
        records.append(prediction_from_state(question, state))
        logger.info("完成评估题目 id=%s", question.id)
    return records


def write_predictions(records: list[PredictionRecord], path: Path) -> None:
    """全部题目运行成功后写 UTF-8 JSONL，每题一行，仅含规定的六字段。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )


def read_predictions(path: Path) -> list[PredictionRecord]:
    """读取程序生成的六字段 JSONL，用于不调用模型的离线复评。"""
    return [
        PredictionRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
