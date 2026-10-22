"""只验证已接线的运行行为，不为 TODO 算法编造成功结果。"""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from search_agent.core import create_search_agent
from tests.test_agent import StreamingModel


def test_incomplete_tool_round_is_removed_only_from_model_view():
    from search_agent.core.context.messages import repair_tool_messages

    history = [
        HumanMessage(content="Search"),
        AIMessage(
            content="Checking",
            tool_calls=[
                {"name": "retrieve", "args": {}, "id": "done"},
                {"name": "retrieve", "args": {}, "id": "lost"},
            ],
        ),
        ToolMessage(content="actual evidence", tool_call_id="done"),
        HumanMessage(content="Continue"),
        ToolMessage(content="orphan", tool_call_id="other"),
    ]
    result = repair_tool_messages(history)
    assert [c["id"] for c in result[1].tool_calls] == ["done"]
    assert [m.type for m in result] == ["human", "ai", "tool", "human"]
    assert len(history[1].tool_calls) == 2
    assert history[2].content == "actual evidence"


def test_repair_does_not_leave_raw_provider_tool_calls():
    from langchain_openai.chat_models.base import _convert_message_to_dict

    from search_agent.core.context.messages import repair_tool_messages

    message = AIMessage(
        content="Checking",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "lost",
                    "type": "function",
                    "function": {"name": "retrieve", "arguments": "{}"},
                }
            ]
        },
    )
    result = repair_tool_messages([message, HumanMessage(content="Continue")])
    # 即使规范化 tool_calls 已清空，OpenAI 适配器也可能读 additional_kwargs。
    assert not _convert_message_to_dict(result[0]).get("tool_calls")
    assert message.additional_kwargs["tool_calls"][0]["id"] == "lost"


def test_registered_context_repairs_history_and_preserves_raw_messages():
    from search_agent.cli.main import stream_turn

    model = StreamingModel(responses=[AIMessage(content="Continued")])
    history = [
        HumanMessage(content="Search"),
        AIMessage(
            content="", tool_calls=[{"name": "retrieve", "args": {"query": "a"}, "id": "lost"}]
        ),
    ]
    result = stream_turn(create_search_agent(model=model), "Continue", history)
    assert not any(isinstance(m, AIMessage) and m.tool_calls for m in model.requests[0])
    assert result[1].tool_calls[0]["id"] == "lost"


def test_todos_survive_next_cli_turn_and_are_visible(capsys):
    from search_agent.cli.main import stream_turn

    todos = [{"content": "Find the source", "status": "in_progress"}]
    model = StreamingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "write_todos", "args": {"todos": todos}, "id": "plan"}],
            ),
            AIMessage(content="Planned"),
            AIMessage(content="Continuing"),
        ]
    )
    state = {}
    agent = create_search_agent(model=model)
    history = stream_turn(agent, "Plan the research", [], state=state)
    assert state["todos"] == todos
    stream_turn(agent, "Continue", history, state=state)
    assert state["todos"] == todos
    assert "Find the source" in capsys.readouterr().out


def test_extended_tool_parameters_reach_placeholder_without_fake_results():
    calls = [
        ("retrieve", {"query": "a", "method": "hybrid", "instruction": "Find dates"}),
        ("read_skill_resource", {"name": "search-recovery", "path": "references/tips.md"}),
        (
            "run_skill_script",
            {"name": "search-recovery", "script": "scripts/rewrite.py", "arguments": ["q"]},
        ),
        ("inspect_document", {"doc_id": "doc-1", "query": "a"}),
    ]
    for name, args in calls:
        model = StreamingModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call"}]),
                AIMessage(content="Unfinished"),
            ]
        )
        result = create_search_agent(model=model).invoke(
            {"messages": [HumanMessage(content="Try")]}
        )
        response = json.loads(result["messages"][2].content)
        assert response["status"] == "not_implemented"
        assert response.get("result") is None


def test_async_agent_with_registered_middleware():
    model = StreamingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": [{"content": "Search", "status": "pending"}]},
                        "id": "plan",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve",
                        "args": {"query": "a", "method": "dense"},
                        "id": "search",
                    }
                ],
            ),
            AIMessage(content="Async works"),
        ]
    )
    result = asyncio.run(
        create_search_agent(model=model).ainvoke({"messages": [HumanMessage(content="Hello")]})
    )
    assert result["messages"][-1].text == "Async works"
    assert result["todos"][0]["content"] == "Search"
    assert json.loads(result["messages"][4].content)["status"] == "not_implemented"


def test_preview_truncation_does_not_change_tool_history(capsys):
    from langchain_core.tools import tool

    from search_agent.cli.display import TerminalView
    from search_agent.cli.main import stream_turn

    @tool
    def long_result() -> str:
        """Return a long test document."""
        return "x" * 2000 + "ORIGINAL_END"

    model = StreamingModel(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "long_result", "args": {}, "id": "long"}]),
            AIMessage(content="Done"),
        ]
    )
    result = stream_turn(
        create_search_agent(model=model, tools=[long_result]),
        "Read",
        [],
        view=TerminalView(plain=True),
    )
    assert result[2].content.endswith("ORIGINAL_END")
    output = capsys.readouterr().out
    assert "ORIGINAL_END" not in output
    assert "--verbose" in output


def test_cli_clear_discards_todos_and_commands_do_not_call_model(monkeypatch, capsys):
    from search_agent.cli.display import TerminalView
    from search_agent.cli.main import chat

    model = StreamingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": [{"content": "First task", "status": "pending"}]},
                        "id": "plan",
                    }
                ],
            ),
            AIMessage(content="Planned"),
        ]
    )
    inputs = iter(["Plan", "/clear", "/todos", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chat(create_search_agent(model=model), view=TerminalView(plain=True))
    assert len(model.requests) == 2
    output = capsys.readouterr().out
    assert "尚未创建任务计划" in output
    assert "API tokens：未统计" in output
