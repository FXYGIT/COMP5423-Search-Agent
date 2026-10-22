"""作业 1.3：运行时技能指令注入的实现位置。"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from search_agent.core.state import SearchState


class SkillMiddleware(AgentMiddleware[SearchState, None]):
    """向主模型提供技能目录，并把选中技能的正文注入当前请求。

    TODO：通过 SkillLoader.discover 提供元数据，让模型自行调用 load_skill。
    根据工具结果设置 active_skill，只加入当前技能的正文。
    TODO：加载后写入 active_skill/skill_instructions；卸载时清空二者。
    当前 hook 透传，不读取技能、不注入伪造的指令。
    """

    state_schema = SearchState

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """在调用 handler 前构造本轮的技能上下文。

        TODO：读取当前技能，将正文追加到请求副本中的 system_message，
        再调用 handler；下一轮重新构造，避免永久修改基础提示词。
        """
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """异步透传入口；TODO：与同步入口共用目录摘要与当前技能的请求构建逻辑。"""
        return await handler(request)
