"""LangChain agent 的唯一组装入口，供 CLI 和未来网关直接调用。"""

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

from search_agent.core.config import get_config
from search_agent.core.middlewares import (
    SearchContextMiddleware,
    SkillMiddleware,
    TodoListMiddleware,
    TrajectoryMiddleware,
)
from search_agent.core.state import SearchState
from search_agent.core.tools import (
    find_in_document,
    inspect_document,
    list_skills,
    load_skill,
    open_document,
    read_skill_resource,
    retrieve,
    run_skill_script,
    unload_skill,
)

SYSTEM_PROMPT = """You are a conversational search assistant for a fixed document corpus.
Use retrieve to search, open_document to read evidence, and find_in_document to locate clues.
You may reformulate queries and call tools repeatedly based on their results.
Use load_skill when a reusable procedure is helpful.
Use list_skills to discover skills, read_skill_resource to read supporting files,
run_skill_script to run a loaded skill's scripts, and unload_skill when finished.
Use inspect_document for query-focused reading within a known document.
Use write_todos for multi-step research; it is functional and tracks progress.
Answer follow-up questions using the conversation history.
The search/document/skill tools are currently scaffolds. If a tool returns not_implemented, explain
that the feature is unfinished; do not repeat it or claim to have retrieved evidence.
When evidence is available, cite document IDs in the answer.
"""


def create_search_agent(
    *,
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """创建最小工具调用 agent，不承担终端输入、输出或历史存储。

    默认使用 .env 配置的 ChatOpenAI、项目工具和 LangChain 原生 TODO 工具。
    CLI/网关跨轮传入完整 SearchState，使用 invoke/ainvoke/stream/astream。
    模型和工具参数用于离线测试，也方便后续接入已实现的业务模块。
    LangChain 负责模型→工具→模型的循环，不再自行包装一套循环。
    middleware 注册后未实现部分透传；仅规划与工具配对有最小实际逻辑。
    """
    if model is None:
        config = get_config()
        model = ChatOpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            streaming=True,
        )
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        SearchContextMiddleware(),
        SkillMiddleware(),
        TodoListMiddleware(),
        TrajectoryMiddleware(),
    ]
    return create_agent(
        model=model,
        tools=list(tools)
        if tools is not None
        else [
            retrieve,
            open_document,
            find_in_document,
            load_skill,
            inspect_document,
            list_skills,
            read_skill_resource,
            run_skill_script,
            unload_skill,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=SearchState,
        middleware=middleware,
    )
