"""使用示例和离线 LangChain 图验证提交协议，不请求真实 API。"""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from tests.test_agent import StreamingModel


def test_example_round_trips_without_changing_schema(tmp_path):
    from search_agent.evaluation.runner import read_predictions, write_predictions

    source = Path(__file__).parent / "fixtures" / "test_prediction.example.jsonl"
    records = read_predictions(source)
    assert records[0].tool_calls == 4
    assert records[0].total_tokens == 4260
    target = tmp_path / "predictions.jsonl"
    write_predictions(records, target)
    assert [json.loads(line) for line in target.read_text().splitlines()] == [
        json.loads(line) for line in source.read_text().splitlines()
    ]


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream"])
def test_real_agent_collects_example_format_and_usage(mode, tmp_path):
    from search_agent.core import create_search_agent
    from search_agent.core.config import AppConfig, LoggingConfig
    from search_agent.evaluation.runner import prediction_from_state, write_predictions
    from search_agent.evaluation.schemas import EvaluationQuestion

    @tool
    def lookup(query: str) -> str:
        """Return an offline test document."""
        return json.dumps({"text": query}, ensure_ascii=False)

    model = StreamingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "lookup", "args": {"query": "望远镜"}},
                    {"id": "c2", "name": "lookup", "args": {"query": "designer"}},
                ],
                usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            AIMessage(
                content="Mira Vale",
                usage_metadata={"input_tokens": 20, "output_tokens": 3, "total_tokens": 23},
            ),
            AIMessage(
                content="Follow-up",
                usage_metadata={"input_tokens": 30, "output_tokens": 4, "total_tokens": 34},
            ),
        ]
    )
    agent = create_search_agent(
        model=model,
        tools=[lookup],
        system_prompt="Answer using tools.",
        config=AppConfig(logging=LoggingConfig(path=tmp_path / "agent.log")),
    )
    initial = {"messages": [HumanMessage(content="Who?")]}
    if mode == "ainvoke":
        state = asyncio.run(agent.ainvoke(initial))
    elif mode == "stream":
        from search_agent.cli.main import stream_turn

        state = {}
        stream_turn(agent, "Who?", [], state=state)
    else:
        state = agent.invoke(initial)
    record = prediction_from_state(EvaluationQuestion(id="a", question="Who?"), state)
    target = tmp_path / "prediction.jsonl"
    write_predictions([record], target)
    row = json.loads(target.read_text(encoding="utf-8"))
    entries = row["trajectory"]
    assert [m["role"] for m in entries] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert "Answer using tools." in entries[0]["content"]
    assert "write_todos" in entries[0]["content"]  # 实际注入后的 system prompt。
    assert entries[2]["content"] is None
    assert entries[2]["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "望远镜"}'},
    }
    assert entries[2]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert entries[3]["tool_call_id"] == "c1"
    assert entries[3]["name"] == "lookup"
    assert json.loads(entries[3]["content"]) == {"text": "望远镜"}
    assert entries[-1]["content"].strip() == row["answer"] == "Mira Vale"
    assert row["tool_calls"] == 2
    assert row["total_tokens"] == 35

    # CLI 跨轮传完整 state 时，历史用量只计算一次。
    state["messages"].append(HumanMessage(content="And?"))
    followup = agent.invoke(state)
    assert followup["tool_calls"] == 2
    assert followup["total_tokens"] == 69
    assert len(followup["trajectory"]) == 8


def test_missing_api_usage_allows_chat_but_blocks_submission(tmp_path):
    from search_agent.core import create_search_agent
    from search_agent.core.config import AppConfig, LoggingConfig
    from search_agent.evaluation.runner import prediction_from_state
    from search_agent.evaluation.schemas import EvaluationQuestion

    agent = create_search_agent(
        model=StreamingModel(responses=[AIMessage(content="Hello")]),
        tools=[],
        config=AppConfig(logging=LoggingConfig(path=tmp_path / "agent.log")),
    )
    state = agent.invoke({"messages": [HumanMessage(content="Hi")]})
    assert state["messages"][-1].text == "Hello"
    assert state["total_tokens"] is None
    with pytest.raises(ValueError, match="usage"):
        prediction_from_state(EvaluationQuestion(id="a", question="Hi"), state)


def test_prediction_rejects_old_events_and_inconsistent_totals():
    from pydantic import ValidationError

    from search_agent.evaluation.schemas import PredictionRecord

    source = Path(__file__).parent / "fixtures" / "test_prediction.example.jsonl"
    row = json.loads(source.read_text().splitlines()[0])
    with pytest.raises(ValidationError):
        PredictionRecord.model_validate({**row, "trajectory": [{"type": "model", "text": "A"}]})
    with pytest.raises(ValidationError, match="tool_calls"):
        PredictionRecord.model_validate({**row, "tool_calls": 1})
    with pytest.raises(ValidationError, match="total_tokens"):
        PredictionRecord.model_validate({**row, "total_tokens": 1})


def test_provider_usage_and_direct_tool_result_are_preserved(tmp_path):
    from search_agent.core import create_search_agent
    from search_agent.core.config import AppConfig, LoggingConfig
    from search_agent.core.trajectory import ToolEntry

    @tool(return_direct=True)
    def lookup() -> str:
        """Return the entire test document without another model round."""
        return "evidence " * 500

    usage = {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
    model = StreamingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"id": "direct", "name": "lookup", "args": {}}],
                response_metadata={"token_usage": usage},
            )
        ]
    )
    agent = create_search_agent(
        model=model,
        tools=[lookup],
        config=AppConfig(logging=LoggingConfig(path=tmp_path / "agent.log")),
    )
    state = agent.invoke({"messages": [HumanMessage(content="Read")]})
    assert state["total_tokens"] == 40
    assert state["tool_calls"] == 1
    assert state["trajectory"][2].usage.model_dump() == usage
    assert isinstance(state["trajectory"][-1], ToolEntry)
    assert state["trajectory"][-1].content == "evidence " * 500


def test_reused_agent_has_independent_per_question_statistics(tmp_path):
    from search_agent.core import create_search_agent
    from search_agent.core.config import AppConfig, LoggingConfig
    from search_agent.evaluation.runner import run_questions
    from search_agent.evaluation.schemas import EvaluationQuestion

    model = StreamingModel(
        responses=[
            AIMessage(
                content="First",
                usage_metadata={
                    "input_tokens": 9,
                    "output_tokens": 1,
                    "total_tokens": 10,
                },
            ),
            AIMessage(
                content="Second",
                usage_metadata={
                    "input_tokens": 19,
                    "output_tokens": 1,
                    "total_tokens": 20,
                },
            ),
        ]
    )
    agent = create_search_agent(
        model=model,
        tools=[],
        config=AppConfig(logging=LoggingConfig(path=tmp_path / "agent.log")),
    )
    records = run_questions(
        agent,
        [
            EvaluationQuestion(id="a", question="Q1", answer="secret"),
            EvaluationQuestion(id="b", question="Q2", answer="secret"),
        ],
    )
    assert [record.total_tokens for record in records] == [10, 20]
    assert [len(record.trajectory) for record in records] == [3, 3]
    assert "Q1" not in records[1].model_dump_json()
    assert all("secret" not in record.model_dump_json() for record in records)
