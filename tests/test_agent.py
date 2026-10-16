"""验证可运行的 CLI/MVP；作业算法骨架不做伪实现测试。"""

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from pydantic import Field

from search_agent.core import create_search_agent


class StreamingModel(FakeMessagesListChatModel):
    requests: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(messages)
        response = self.responses[self.i]
        self.i += 1
        if response.tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_calls=response.tool_calls)
            )
        else:
            for word in response.text.split(" "):
                yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def test_stream_turn_prints_text_and_tools_before_execution(capsys):
    from search_agent.cli.main import stream_turn

    observed = []

    @tool
    def lookup(query: str) -> str:
        """Read a test fact."""
        observed.append(capsys.readouterr().out)
        return "Alice"

    model = StreamingModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "lookup", "args": {"query": "winner"}, "id": "c1"}]
            ),
            AIMessage(content="Alice won"),
        ]
    )
    agent = create_search_agent(model=model, tools=[lookup])
    history = stream_turn(agent, "Who won?", [])
    output = "".join(observed) + capsys.readouterr().out
    assert "lookup" in observed[0]
    assert "winner" in observed[0]
    assert "Alice" in output
    assert output.count("Alice won") == 1
    assert [m.type for m in history] == ["human", "ai", "tool", "ai"]
    assert isinstance(history[2], ToolMessage)


def test_followup_sends_prior_turn_to_model():
    from search_agent.cli.main import stream_turn

    model = StreamingModel(
        responses=[AIMessage(content="Hello"), AIMessage(content="Your name is Alice")]
    )
    agent = create_search_agent(model=model, tools=[])
    history = stream_turn(agent, "My name is Alice", [])
    history = stream_turn(agent, "What is my name?", history)
    assert [m.content for m in model.requests[1] if isinstance(m, HumanMessage)] == [
        "My name is Alice",
        "What is my name?",
    ]
    assert len(history) == 4


def test_cli_clear_discards_history(monkeypatch, capsys):
    from search_agent.cli.main import chat

    model = StreamingModel(responses=[AIMessage(content="First"), AIMessage(content="Second")])
    agent = create_search_agent(model=model, tools=[])
    inputs = iter(["Remember me", "/clear", "New question", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chat(agent)
    assert [m.content for m in model.requests[1] if isinstance(m, HumanMessage)] == ["New question"]
    assert "清空" in capsys.readouterr().out


def test_default_placeholder_tool_does_not_claim_real_search(capsys):
    from search_agent.cli.main import stream_turn

    model = StreamingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "retrieve", "args": {"query": "winner"}, "id": "c1"}],
            ),
            AIMessage(content="Retrieval is not implemented"),
        ]
    )
    history = stream_turn(create_search_agent(model=model), "Search for winner", [])
    assert "not_implemented" in history[2].content
    assert "not_implemented" in capsys.readouterr().out


def test_text_is_printed_while_model_is_still_streaming(monkeypatch):
    from threading import Event

    from search_agent.cli.main import stream_turn

    displayed = Event()
    original_print = print

    def observe_print(*args, **kwargs):
        original_print(*args, **kwargs)
        if args == ("First",):
            displayed.set()

    monkeypatch.setattr("builtins.print", observe_print)

    class ObservedStream(StreamingModel):
        def _stream(self, messages, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="First"))
            # Graph 在工作线程中生成 token，等待终端消费首段后才生成下一段。
            assert displayed.wait(timeout=3), "CLI buffered text until model completion"
            yield ChatGenerationChunk(message=AIMessageChunk(content=" second"))

    agent = create_search_agent(model=ObservedStream(responses=[]), tools=[])
    stream_turn(agent, "Hello", [])
    assert displayed.is_set()
