"""作业 1.3：运行时技能指令注入的实现位置。"""

from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from search_agent.core.state import SearchState


class SkillMiddleware(AgentMiddleware[SearchState, None]):
    """向主模型提供技能目录，并把选中技能的正文注入当前请求。

    TODO：通过 SkillLoader.discover 提供元数据，让模型自行调用 load_skill。
    根据工具结果设置 active_skill，只加入当前技能的正文。
    不设计独立技能管理器或会话系统。本类当前不注册。
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
        raise NotImplementedError
