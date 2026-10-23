"""记录主 Agent 的实际消息轨迹；数据格式由 core.trajectory 统一定义。"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ExtendedModelResponse
from langgraph.runtime import Runtime
from langgraph.types import Command

from search_agent.core.state import SearchState
from search_agent.core.trajectory import SystemEntry, message_to_entry, trajectory_totals

logger = logging.getLogger(__name__)


class TrajectoryMiddleware(AgentMiddleware[SearchState, None]):
    """把新增原始消息与完整模型响应写入 state，供 CLI/评估复用。

    放在模型请求包装链的最内层，读取技能/规划注入后的真实 system prompt。
    统计按 state 保存，实例本身不保存会话数据，跨题不会互相污染。
    message_count 仅记录已处理的原始消息数，避免多轮对话重复计费。
    TODO：摘要/技能辅助 LLM 接通时，在调用位置追加同格式消息与真实 usage；
    不可只增加 total_tokens 而漏掉对应轨迹。当前仅覆盖主 Agent 的调用。
    """

    state_schema = SearchState

    def record_response(
        self, request: ModelRequest[None], response: ModelResponse[Any]
    ) -> ExtendedModelResponse[Any]:
        """保留原始历史及 API 响应；工具配对/摘要后的请求视图不替换原始轨迹。"""
        state = cast(SearchState, request.state)
        entries = list(state.get("trajectory", []))
        count = state.get("trajectory_message_count", 0)
        pending = [message_to_entry(message) for message in state["messages"][count:]]
        # 首次请求以 system 开始；后续先记录到达的消息，再记录改变的 system。
        if entries:
            entries.extend(pending)
            pending = []
        # 静态 system 只写一次；未来技能改变 prompt 时记录新的系统消息。
        if request.system_message is not None:
            system = message_to_entry(request.system_message)
            previous = next(
                (item for item in reversed(entries) if isinstance(item, SystemEntry)), None
            )
            if system != previous:
                entries.append(system)
        entries.extend(pending)
        entries.extend(message_to_entry(message) for message in response.result)
        calls, tokens = trajectory_totals(entries)
        logger.debug("模型步骤完成，累计工具调用=%d，API tokens=%s", calls, tokens)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(
                update={
                    "trajectory": entries,
                    "trajectory_message_count": len(state["messages"]) + len(response.result),
                    "tool_calls": calls,
                    "total_tokens": tokens,
                }
            ),
        )

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ExtendedModelResponse[Any]:
        """同步调用返回完整响应后记录一次，不逐 chunk 计数。"""
        return self.record_response(request, handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ExtendedModelResponse[Any]:
        """异步与流式调用共享同一记录逻辑。"""
        return self.record_response(request, await handler(request))

    def after_agent(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """补录结束前尚未经过下一次模型调用的消息，例如 return_direct 工具结果。"""
        count = state.get("trajectory_message_count", 0)
        entries = [
            *state.get("trajectory", []),
            *(message_to_entry(m) for m in state["messages"][count:]),
        ]
        calls, tokens = trajectory_totals(entries)
        logger.info("Agent 本轮完成，累计工具调用=%d，API tokens=%s", calls, tokens)
        return {
            "trajectory": entries,
            "trajectory_message_count": len(state["messages"]),
            "tool_calls": calls,
            "total_tokens": tokens,
        }

    async def aafter_agent(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """异步收尾与同步保持一致。"""
        return self.after_agent(state, runtime)
