"""作业 1.2 Feature B：结构化搜索状态及模型上下文控制的实现位置。"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.runtime import Runtime

from search_agent.core.context.messages import repair_tool_messages
from search_agent.core.state import SearchState


class SearchContextMiddleware(AgentMiddleware[SearchState, None]):
    """整理工具结果，为下一轮模型调用提供搜索进度与选中证据。

    TODO：记录已访问文档、证据和当前子查询；去重或裁剪传给模型的上下文。
    原始 messages 留给轨迹统计，不直接删除原始工具结果。
    当前仅实现模型请求副本的工具消息配对，其余业务逻辑透传。
    """

    state_schema = SearchState

    def before_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """在下一次推理前，读取新增工具结果并返回 SearchState 的更新字段。

        TODO：提取 doc_id/证据，更新 active_subqueries/candidate_answer。
        TODO：接 context.selection.extract_search_progress；达到预算阈值时调用
        ConversationSummarizer，更新 summary/summarized_message_count，并记录摘要用量。
        当前返回空更新，不生成假进度或摘要。
        """
        return {}

    async def abefore_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """异步调用保持相同骨架行为；TODO：摘要实现后使用模型的异步接口。"""
        return self.before_model(state, runtime)

    def prepare_request(self, request: ModelRequest[None]) -> ModelRequest[None]:
        """只修改发往模型的消息视图，原始 state.messages 保留用于审计。

        TODO：先用 select_context 选择近期消息与证据、注入摘要，再执行工具配对。
        当前不裁剪历史、不进行摘要，因此不能视为完整的作业上下文管理实现。
        """
        return request.override(messages=repair_tool_messages(request.messages))

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """同步模型请求：工具配对后交回 LangChain 执行。"""
        return handler(self.prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """异步模型请求使用同一上下文视图。"""
        return await handler(self.prepare_request(request))
