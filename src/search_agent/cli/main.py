"""完整的流式 CLI：保存对话历史，展示模型文本及工具调用过程。"""

import argparse
import json
from collections.abc import Iterator
from typing import Any, cast

from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from search_agent.core import create_search_agent


def stream_turn(
    agent: CompiledStateGraph[Any, Any, Any, Any],
    question: str,
    history: list[AnyMessage],
) -> list[AnyMessage]:
    """执行一轮对话，实时输出并返回包含本轮完整消息的新历史。

    messages 流用于逐段打印模型文本；updates 流用于打印完整工具参数、
    工具返回结果并收集历史。这样不会把同一回答再打印一遍，
    也不会把尚未拼完的工具参数当成 JSON 解析。
    """
    messages = [*history, HumanMessage(content=question)]
    text_started = False
    events = cast(
        Iterator[tuple[str, Any]],
        agent.stream({"messages": messages}, stream_mode=["messages", "updates"]),
    )
    for mode, data in events:
        if mode == "messages":
            chunk, _ = data
            if isinstance(chunk, (AIMessage, AIMessageChunk)) and chunk.text:
                if not text_started:
                    print("\n助手：", end="", flush=True)
                    text_started = True
                print(chunk.text, end="", flush=True)
        elif mode == "updates":
            for update in data.values():
                for message in update.get("messages", []):
                    messages.append(message)
                    if isinstance(message, AIMessage):
                        if text_started:
                            print(flush=True)
                        elif message.text:
                            print(f"\n助手：{message.text}", flush=True)
                        text_started = False
                        for call in message.tool_calls:
                            args = json.dumps(call["args"], ensure_ascii=False)
                            print(f"\n[调用工具] {call['name']} {args}", flush=True)
                    elif isinstance(message, ToolMessage):
                        print(f"[工具结果] {message.name}\n{message.content}", flush=True)
    return messages


def chat(agent: CompiledStateGraph[Any, Any, Any, Any]) -> None:
    """持续读取输入，并把上一轮模型和工具消息带入下一轮。

    /clear 开始新对话；exit、quit、/exit 或 Ctrl+D/Ctrl+C 退出。
    CLI 不再把每次输入都当成互不相关的评估题。
    """
    history: list[AnyMessage] = []
    print("搜索助手：支持连续追问；/clear 清空对话；exit 退出。")
    while True:
        try:
            question = input("\n你：").strip()
            if question.lower() in {"exit", "quit", "/exit"}:
                break
            if question == "/clear":
                history = []
                print("已清空对话。")
                continue
            if question:
                history = stream_turn(agent, question, history)
        except (EOFError, KeyboardInterrupt):
            break
    print("\n再见！")


def main() -> None:
    """解析可选单次问题，创建一个 agent，再启动流式输出或多轮聊天。"""
    parser = argparse.ArgumentParser(description="LangChain 流式对话 CLI")
    parser.add_argument("question", nargs="?", help="单次提问；省略则进入多轮聊天")
    args = parser.parse_args()
    agent = create_search_agent()
    if args.question:
        stream_turn(agent, args.question, [])
    else:
        chat(agent)


if __name__ == "__main__":
    main()
