"""完整的流式 CLI：保存对话历史，展示模型文本及工具调用过程。"""

import argparse
from collections.abc import Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from search_agent.core import create_search_agent
from search_agent.core.state import SearchState

from .display import TerminalView


def stream_turn(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    question: str,
    history: list[AnyMessage],
    *,
    state: SearchState | None = None,
    view: TerminalView | None = None,
) -> list[AnyMessage]:
    """执行一轮对话，实时输出并返回包含本轮完整消息的新历史。

    messages 流显示文本，updates 显示工具过程，values 保存完整最新状态。
    state 就地更新，调用方可跨轮保留 todos/summary/skills；旧 history 参数仍可用。
    不把 chunks 或 middleware 的空更新当成完整消息重复追加。
    """
    state = state if state is not None else SearchState(messages=[])
    view = view or TerminalView()
    state["messages"] = [*history, HumanMessage(content=question)]
    previous_todos = state.get("todos", [])
    previous_skill = state.get("active_skill")
    previous_evidence = state.get("selected_evidence", [])
    events = cast(
        Iterator[tuple[str, Any]],
        agent.stream(state, stream_mode=["messages", "updates", "values"]),
    )
    for mode, data in events:
        if mode == "messages":
            chunk, _ = data
            if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.text:
                view.token(chunk.text)
        elif mode == "updates":
            for update in data.values():
                if not isinstance(update, dict):
                    continue
                for message in update.get("messages", []):
                    if isinstance(message, AIMessage):
                        if view.text_started:
                            view.end_text()
                        elif message.text:
                            view.token(message.text)
                            view.end_text()
                        for call in message.tool_calls:
                            view.tool_call(call["name"], call["args"])
                    elif isinstance(message, ToolMessage):
                        view.tool_result(
                            message.name or message.tool_call_id,
                            message.text,
                            failed=message.status == "error",
                        )
        elif mode == "values":
            state.update(data)
            if state.get("todos", []) != previous_todos:
                view.todos(state)
                previous_todos = state.get("todos", [])
            if state.get("active_skill") != previous_skill:
                view.skills(state)
                previous_skill = state.get("active_skill")
            if state.get("selected_evidence", []) != previous_evidence:
                view.evidence(state)
                previous_evidence = state.get("selected_evidence", [])
    view.end_text()
    return state["messages"]


def chat(
    agent: CompiledStateGraph[Any, Any, Any, Any], *, view: TerminalView | None = None
) -> None:
    """持续读取输入，并把上一轮模型和工具消息带入下一轮。

    /clear 开始新对话；exit、quit、/exit 或 Ctrl+D/Ctrl+C 退出。
    CLI 不再把每次输入都当成互不相关的评估题。
    """
    state = SearchState(messages=[])
    view = view or TerminalView()
    view.welcome()
    while True:
        try:
            question = input("\n你：").strip()
            if question.lower() in {"exit", "quit", "/exit"}:
                break
            if question == "/clear":
                state = SearchState(messages=[])
                view.panel("新会话", "已清空对话、任务、技能、摘要和证据。")
                continue
        except (EOFError, KeyboardInterrupt):
            break
        commands = {
            "/todos": view.todos,
            "/skills": view.skills,
            "/evidence": view.evidence,
            "/status": view.status,
        }
        if question == "/help":
            view.help()
        elif question in commands:
            commands[question](state)
        elif question:
            try:
                stream_turn(agent, question, state.get("messages", []), state=state, view=view)
            except KeyboardInterrupt:
                view.end_text()
                view.panel("已中断", "可继续输入；下一轮会处理未完成的工具调用。", "yellow")
            except Exception as exc:
                view.end_text()
                view.panel("本轮失败", f"{type(exc).__name__}: {exc}", "red")
    print("\n再见！")


def main() -> None:
    """解析可选单次问题，创建一个 agent，再启动流式输出或多轮聊天。"""
    parser = argparse.ArgumentParser(description="LangChain 流式对话 CLI")
    parser.add_argument("question", nargs="?", help="单次提问；省略则进入多轮聊天")
    parser.add_argument("--plain", action="store_true", help="纯文本输出，关闭面板和颜色")
    parser.add_argument("--verbose", action="store_true", help="显示完整工具结果")
    args = parser.parse_args()
    agent = create_search_agent()
    view = TerminalView(plain=args.plain, verbose=args.verbose)
    if args.question:
        stream_turn(agent, args.question, [], view=view)
    else:
        chat(agent, view=view)


if __name__ == "__main__":
    main()
