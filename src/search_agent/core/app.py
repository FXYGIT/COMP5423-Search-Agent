from __future__ import annotations

import os
from typing import Any

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import SecretStr

load_dotenv()


class CoreApp:
    def __init__(self) -> None:
        pass

    async def run(self) -> None:
        pass

    # ------------------------------------------------------- 演示代码 👇
    def get_agent(self) -> Any:
        model = ChatOpenAI(
            api_key=SecretStr(os.getenv("LLM_API_KEY", "")),
            base_url=os.getenv("LLM_BASE_URL", ""),
            model=os.getenv("LLM_MODEL_NAME", ""),
        )

        search_tool = TavilySearch(
            max_results=5,
            search_depth="advanced",
        )

        agent = create_deep_agent(
            model=model,
            tools=[search_tool],
            system_prompt="You are a helpful assistant",
        )

        return agent


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        agent = CoreApp().get_agent()
        messages: list[Any] = []

        print("输入 exit 或 quit 退出对话。")

        while True:
            try:
                user_input = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if user_input.lower() in {"exit", "quit"}:
                print("再见！")
                break

            if not user_input:
                continue

            try:
                response = await agent.ainvoke(
                    {
                        "messages": [
                            *messages,
                            {"role": "user", "content": user_input},
                        ]
                    },
                    config={"configurable": {"thread_id": "1"}},
                )
                messages = response["messages"]
                print(f"\n助手：{messages[-1].content}")
            except Exception as exc:
                print(f"\n请求失败：{exc}")

    asyncio.run(main())
