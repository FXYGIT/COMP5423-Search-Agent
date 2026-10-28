"""B 模块离线验证：真实图、工具协议、预算、摘要轨迹及会话隔离。"""

import asyncio
import json

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import ValidationError

from search_agent.core import create_search_agent
from search_agent.core.config import AppConfig, ContextConfig, LlmConfig, LoggingConfig
from search_agent.core.context.budget import estimate_text_tokens, estimate_tokens
from search_agent.core.context.selection import (
    deduplicate_evidence,
    extract_search_progress,
    select_context,
)
from search_agent.core.context.summary import ConversationSummarizer
from search_agent.core.middlewares.context import SearchContextMiddleware
from search_agent.core.trajectory import AssistantEntry, ToolEntry, trajectory_totals
from tests.test_agent import StreamingModel


def usage(tokens):
    return {"input_tokens": tokens - 1, "output_tokens": 1, "total_tokens": tokens}


def page(doc_id="d1", text="Alice won.", start=0):
    return {
        "doc_id": doc_id,
        "text": text,
        "start": start,
        "end": start + len(text),
        "source_url": "https://example.org/" + doc_id,
    }


def exchange(name, result, *, call_id="c1", args=None, status="ok"):
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": call_id,
                    "name": name,
                    "args": args or {"doc_id": "d1"},
                }
            ],
        ),
        ToolMessage(
            content=json.dumps({"status": status, "result": result}),
            name=name,
            tool_call_id=call_id,
        ),
    ]


def context_config(**overrides):
    return ContextConfig(
        **{
            "max_tokens": 6000,
            "reserve_output_tokens": 256,
            "summary_max_tokens": 256,
            "summary_trigger_tokens": 512,
            **overrides,
        }
    )


class ObservedModel(StreamingModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class AsyncOnlySummary(ObservedModel):
    async_calls: int = 0

    def _generate(self, messages, **kwargs):
        raise AssertionError("Async agent must not call the synchronous summary API")

    async def _agenerate(self, messages, **kwargs):
        self.async_calls += 1
        self.requests.append(messages)
        response = self.responses[self.i % len(self.responses)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def test_extracts_only_original_successful_document_evidence_and_is_idempotent():
    messages = [
        HumanMessage(content="Who won?"),
        *exchange("retrieve", [{"doc_id": "d0", "snippet": "Unlocated"}], args={"query": "winner"}),
        *exchange("open_document", page()),
        *exchange("open_document", page(), call_id="c2"),
        *exchange("open_document", page("unimplemented"), status="not_implemented"),
        *exchange("load_skill", page()),
        *exchange(
            "inspect_document",
            {
                "doc_id": "d1",
                "query": "winner",
                "passages": [page(text="won.", start=6)],
            },
            args={"doc_id": "d1", "query": "winner"},
        ),
        *exchange(
            "find_in_document",
            {
                "doc_id": "d1",
                "matches": [
                    {
                        "context": "Alice won.",
                        "context_start": 0,
                        "context_end": 10,
                    }
                ],
            },
        ),
        AIMessage(content="Alice"),
    ]
    original = [m.model_dump() for m in messages]
    state = {"messages": messages}
    update = extract_search_progress(state)
    assert update["visited_doc_ids"] == ["d1"]
    assert [e.model_dump() for e in update["selected_evidence"]] == [page()]
    assert update["active_subqueries"] == ["winner"]
    assert update["candidate_answer"] == "Alice"
    state.update(update)
    assert extract_search_progress(state) == update
    assert original == [m.model_dump() for m in messages]
    state["messages"] = [*messages, HumanMessage(content="What year?")]
    assert extract_search_progress(state)["candidate_answer"] is None


@pytest.mark.parametrize("kind", ["orphan", "mismatch", "bad_json", "error", "bad_offset"])
def test_rejects_invalid_or_unpaired_tool_evidence(kind):
    messages = exchange("open_document", page())
    if kind == "orphan":
        messages = messages[1:]
    elif kind == "mismatch":
        messages[1].tool_call_id = "other"
    elif kind == "bad_json":
        messages[1].content = "not json"
    elif kind == "error":
        messages[1].status = "error"
    else:
        messages[1].content = json.dumps({"status": "ok", "result": {**page(), "end": 500}})
    assert extract_search_progress({"messages": messages})["selected_evidence"] == []


def test_artifacts_and_todos_are_consumed_without_planning_model():
    messages = exchange("open_document", None)
    messages[1].artifact = {"status": "ok", "result": page()}
    state = {
        "messages": messages,
        "todos": [
            {"content": "Read source", "status": "completed"},
            {"content": "Verify date", "status": "in_progress"},
        ],
    }
    progress = extract_search_progress(state)
    assert progress["active_subqueries"] == ["Verify date"]
    assert progress["selected_evidence"][0].text == "Alice won."


def test_dedup_merges_matching_overlap_but_preserves_conflicting_originals():
    values = [page(text="abcdef"), page(text="defghi", start=3), page(text="bc", start=1)]
    result = deduplicate_evidence(values)
    assert len(result) == 1
    assert (result[0].start, result[0].end, result[0].text) == (0, 9, "abcdefghi")
    assert len(deduplicate_evidence([*result, page(text="XYZ", start=2)])) == 2


def test_budget_preserves_latest_question_and_tool_pairs_without_mutating_history():
    messages = [
        HumanMessage(content="Old question"),
        AIMessage(content="old " * 3000),
        HumanMessage(content="Latest question"),
        *exchange("open_document", page(text="证据 " * 5000)),
    ]
    original = [m.model_dump() for m in messages]
    state = {"messages": messages, "summary": "Known fact from d1", "summarized_message_count": 2}
    selected = select_context(state, 400, config=context_config())
    assert estimate_tokens(selected) <= 400
    assert any(m.content == "Latest question" for m in selected)
    caller = next(m for m in selected if isinstance(m, AIMessage))
    result = next(m for m in selected if isinstance(m, ToolMessage))
    assert caller.tool_calls[0]["id"] == result.tool_call_id == "c1"
    assert "truncated" in result.content
    assert [m.model_dump() for m in messages] == original


def test_request_budget_includes_system_and_tool_schemas():
    @tool
    def lookup(query: str) -> str:
        """An offline search tool."""
        return query

    config = context_config(summary_enabled=False)
    messages = [
        HumanMessage(content="Question"),
        *exchange("open_document", page(text="x" * 30000)),
    ]
    request = ModelRequest(
        model=ObservedModel(responses=[]),
        messages=messages,
        system_message=SystemMessage(content="Instructions " * 50),
        tools=[lookup],
        state={"messages": messages},
    )
    prepared = SearchContextMiddleware(config).prepare_request(request)
    from langchain_core.utils.function_calling import convert_to_openai_tool

    overhead = (
        estimate_tokens([prepared.system_message])
        + estimate_text_tokens(
            json.dumps([convert_to_openai_tool(lookup)], ensure_ascii=False),
        )
        + 32
    )
    assert estimate_tokens(prepared.messages) + overhead + config.reserve_output_tokens <= 6000
    assert prepared.model_settings["max_tokens"] == config.reserve_output_tokens
    assert request.messages[2].content == messages[2].content


def test_oversized_latest_question_fails_clearly_instead_of_silently_losing_it():
    with pytest.raises(ValueError, match="预算"):
        select_context({"messages": [HumanMessage(content="x" * 10000)]}, 100)
    with pytest.raises(ValidationError):
        context_config(max_tokens=256, reserve_output_tokens=256)


def test_old_rounds_are_selected_whole_and_recent_messages_keep_their_order():
    messages = [
        HumanMessage(content="Huge old question " * 500),
        AIMessage(content="Old answer"),
        HumanMessage(content="Latest"),
        *exchange("open_document", page()),
        *exchange("open_document", page("d2"), call_id="c2", args={"doc_id": "d2"}),
    ]
    result = select_context({"messages": messages}, 700, config=context_config())
    assert [m.type for m in result] == ["human", "ai", "tool", "ai", "tool"]
    assert result[0].content == "Latest"
    assert result[1].tool_calls[0]["id"] == "c1"
    assert result[3].tool_calls[0]["id"] == "c2"


def test_factory_forwards_provider_model_and_runtime_configuration(monkeypatch, tmp_path):
    received = {}

    def create_model(**kwargs):
        received.update(kwargs)
        return ObservedModel(responses=[])

    monkeypatch.setattr("search_agent.core.agent.ChatOpenAI", create_model)
    create_search_agent(
        tools=[],
        config=AppConfig(
            llm=LlmConfig(
                api_key="offline-key",
                base_url="https://example.org/v1",
                model="flash-id",
                timeout_seconds=45,
                max_retries=1,
            ),
            context=context_config(),
            logging=LoggingConfig(path=tmp_path / "agent.log"),
        ),
    )
    assert received["model"] == "flash-id"
    assert received["base_url"] == "https://example.org/v1"
    assert received["timeout"] == 45 and received["max_retries"] == 1
    assert received["stream_usage"] is True and received["max_tokens"] == 256


@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
def test_agent_search_summary_usage_and_independent_states(mode, tmp_path):
    @tool
    def open_document(doc_id: str) -> dict:
        """Return an original offline document."""
        return {"status": "ok", "result": page(doc_id, "Alice won. " * 500)}

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "open_document",
                    "args": {"doc_id": "d1"},
                    "id": "a",
                }
            ],
            usage_metadata=usage(10),
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "open_document",
                    "args": {"doc_id": "d2"},
                    "id": "b",
                }
            ],
            usage_metadata=usage(20),
        ),
        AIMessage(content="Alice", usage_metadata=usage(30)),
        AIMessage(content="New answer", usage_metadata=usage(40)),
    ]
    model = ObservedModel(responses=responses)
    summary_type = AsyncOnlySummary if mode in {"ainvoke", "astream"} else ObservedModel
    summary = summary_type(
        responses=[
            AIMessage(
                content="SUMMARY_PRIVATE: d1 supports Alice; verify d2. Candidate UNVERIFIED.",
                usage_metadata=usage(7),
            )
        ]
    )
    agent = create_search_agent(
        model=model,
        summary_model=summary,
        tools=[open_document],
        system_prompt="Search and verify.",
        config=AppConfig(
            context=context_config(), logging=LoggingConfig(path=tmp_path / "agent.log")
        ),
    )
    initial = {"messages": [HumanMessage(content="Who won?")]}
    if mode == "ainvoke":
        state = asyncio.run(agent.ainvoke(initial))
    elif mode == "astream":

        async def collect():
            return [s async for s in agent.astream(initial, stream_mode="values")][-1]

        state = asyncio.run(collect())
    elif mode == "stream":
        state = list(agent.stream(initial, stream_mode="values"))[-1]
    else:
        state = agent.invoke(initial)
    assert state["visited_doc_ids"] == ["d1", "d2"]
    assert state["candidate_answer"] == "Alice"
    assert len(state["messages"]) == 6
    assert state["summarized_message_count"] == 3
    assert state["tool_calls"] == 2
    assert state["total_tokens"] == 67
    assert trajectory_totals(state["trajectory"]) == (2, 67)
    assert len([e for e in state["trajectory"] if isinstance(e, AssistantEntry)]) == 4
    assert [e.tool_call_id for e in state["trajectory"] if isinstance(e, ToolEntry)] == ["a", "b"]
    assert "SUMMARY_PRIVATE" not in state["messages"][-1].text
    assert len(summary.requests) == 1
    if isinstance(summary, AsyncOnlySummary):
        assert summary.async_calls == 1
    second = agent.invoke({"messages": [HumanMessage(content="New independent question")]})
    assert second["visited_doc_ids"] == []
    assert second["selected_evidence"] == []
    assert second["total_tokens"] == 40
    assert "summary" not in second


def test_summary_is_hidden_from_cli_stream_but_recorded_in_trace(tmp_path, capsys):
    from search_agent.cli.main import stream_turn

    model = ObservedModel(responses=[AIMessage(content="Visible final", usage_metadata=usage(10))])
    summary = ObservedModel(
        responses=[AIMessage(content="PRIVATE_SUMMARY", usage_metadata=usage(7))]
    )
    agent = create_search_agent(
        model=model,
        summary_model=summary,
        tools=[],
        config=AppConfig(
            context=context_config(), logging=LoggingConfig(path=tmp_path / "agent.log")
        ),
    )
    state = {}
    stream_turn(
        agent,
        "New question",
        [
            HumanMessage(content="Old question"),
            AIMessage(content="Long old answer " * 300),
        ],
        state=state,
    )
    output = capsys.readouterr().out
    assert "Visible final" in output
    assert "PRIVATE_SUMMARY" not in output
    assert state["summary"] == "PRIVATE_SUMMARY"
    # 原始旧回答没有 usage，保留未知状态，不能因为摘要有 usage 就伪造完整统计。
    assert state["total_tokens"] is None


def test_incremental_summary_batches_fit_and_do_not_reprocess_old_prefix(tmp_path):
    config = context_config(max_tokens=2500)
    model = ObservedModel(responses=[AIMessage(content="Memory d1", usage_metadata=usage(5))])
    summarizer = ConversationSummarizer(model, config)
    messages = []
    for i in range(4):
        messages.extend([HumanMessage(content=f"Q{i}"), AIMessage(content="fact " * 700)])
    messages.append(HumanMessage(content="Latest"))
    state = {"messages": messages}
    middleware = SearchContextMiddleware(config, summarizer)
    update = middleware.before_model(state, None)
    assert update["summarized_message_count"] == len(messages) - 1
    assert len(model.requests) > 1
    assert all(
        estimate_tokens(r) + config.summary_max_tokens <= config.max_tokens for r in model.requests
    )
    state.update(update)
    count = len(model.requests)
    middleware.before_model(state, None)
    assert len(model.requests) == count


def test_invalid_summary_response_preserves_cursor_and_records_actual_usage():
    config = context_config()
    model = ObservedModel(responses=[AIMessage(content="", usage_metadata=usage(7))])
    middleware = SearchContextMiddleware(config, ConversationSummarizer(model, config))
    state = {
        "messages": [
            HumanMessage(content="Old"),
            AIMessage(content="known " * 500),
            HumanMessage(content="New"),
        ]
    }
    update = middleware.before_model(state, None)
    assert update["summarized_message_count"] == 0
    assert update["summary"] == ""
    assert update["trajectory"][-1].usage.total_tokens == 7


def test_return_direct_document_still_updates_search_state(tmp_path):
    @tool(return_direct=True)
    def open_document(doc_id: str) -> dict:
        """Read a document without another model step."""
        return {"status": "ok", "result": page(doc_id)}

    agent = create_search_agent(
        model=ObservedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "open_document",
                            "args": {"doc_id": "d1"},
                            "id": "a",
                        }
                    ],
                )
            ]
        ),
        tools=[open_document],
        config=AppConfig(logging=LoggingConfig(path=tmp_path / "agent.log")),
    )
    result = agent.invoke({"messages": [HumanMessage(content="Read")]})
    assert result["visited_doc_ids"] == ["d1"]
    assert result["selected_evidence"][0].text == "Alice won."


@pytest.mark.parametrize("asynchronous", [False, True])
def test_candidate_recording_and_native_todo_progress_reach_next_model(asynchronous, tmp_path):
    todos = [{"content": "Verify Alice in source", "status": "in_progress"}]
    model = ObservedModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": todos},
                        "id": "p",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "record_candidate_answer",
                        "args": {"candidate_answer": "Alice"},
                        "id": "c",
                    }
                ],
            ),
            AIMessage(content="Still unverified"),
        ]
    )
    agent = create_search_agent(
        model=model,
        tools=[],
        config=AppConfig(logging=LoggingConfig(path=tmp_path / "agent.log")),
    )
    initial = {"messages": [HumanMessage(content="Find the winner")]}
    if asynchronous:
        result = asyncio.run(agent.ainvoke(initial))
    else:
        result = agent.invoke(initial)
    assert result["active_subqueries"] == ["Verify Alice in source"]
    assert any("UNVERIFIED candidate: Alice" in m.text for m in model.requests[-1])
    assert result["selected_evidence"] == []
