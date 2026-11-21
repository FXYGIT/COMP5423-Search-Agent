"""作业 1.2：结构化搜索状态、预算控制及真实增量摘要。"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.runtime import Runtime

from search_agent.core.config import ContextConfig, get_config
from search_agent.core.context.budget import estimate_text_tokens, estimate_tokens
from search_agent.core.context.messages import repair_tool_messages
from search_agent.core.context.selection import extract_search_progress, select_context
from search_agent.core.context.summary import ConversationSummarizer, SummaryResult
from search_agent.core.state import SearchState
from search_agent.core.trajectory import message_to_entry, trajectory_totals

logger = logging.getLogger(__name__)


class SearchContextMiddleware(AgentMiddleware[SearchState, None]):
    """所有会话数据放入 state；实例仅保存配置及无会话状态的摘要器。"""

    state_schema = SearchState

    def __init__(
        self,
        config: ContextConfig | None = None,
        summarizer: ConversationSummarizer | None = None,
    ) -> None:
        self.config = config or get_config().context
        self.summarizer = summarizer

    def _summary_cutoff(self, state: SearchState) -> int:
        """先压缩完整旧轮次；单轮长搜索则压缩最新完整工具块之前的前缀。"""
        messages = state.get("messages", [])
        cursor = state.get("summarized_message_count", 0)
        size = estimate_tokens(messages[cursor:]) + estimate_text_tokens(state.get("summary", ""))
        if not self.config.summary_enabled or size < self.config.summary_trigger_tokens:
            return cursor
        latest_user = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
            0,
        )
        if latest_user > cursor:
            return latest_user
        latest_ai = next(
            (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], AIMessage)),
            0,
        )
        # 仅有最新问题 + 一个工具块时，暂不为问题本身发起一次无益摘要。
        return latest_ai if latest_ai > max(cursor, latest_user + 1) else cursor

    @staticmethod
    def _record_summary(state: SearchState, result: SummaryResult) -> None:
        # 先补录新到达的原始工具消息，再记录辅助调用；主游标只按原始消息推进。
        messages = state.get("messages", [])
        entries = [
            *state.get("trajectory", []),
            *(message_to_entry(m) for m in messages[state.get("trajectory_message_count", 0) :]),
            *result.trajectory,
        ]
        calls, tokens = trajectory_totals(entries)
        state["summary"] = result.summary
        state["summarized_message_count"] = (
            state.get("summarized_message_count", 0) + result.summarized_message_count
        )
        state["trajectory"] = entries
        state["trajectory_message_count"] = len(messages)
        state["tool_calls"], state["total_tokens"] = calls, tokens

    @staticmethod
    def _updates(state: SearchState) -> dict[str, Any]:
        keys = (
            "visited_doc_ids",
            "selected_evidence",
            "active_subqueries",
            "candidate_answer",
            "search_message_count",
            "summary",
            "summarized_message_count",
            "trajectory",
            "trajectory_message_count",
            "tool_calls",
            "total_tokens",
        )
        values = cast(dict[str, Any], state)
        return {key: values[key] for key in keys if key in values}

    def before_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """处理增量状态，按预算分批压缩旧前缀；不改动原始 messages。"""
        working = cast(SearchState, {**state, **extract_search_progress(state)})
        cutoff = self._summary_cutoff(working)
        while self.summarizer is not None and working.get("summarized_message_count", 0) < cutoff:
            cursor = working.get("summarized_message_count", 0)
            result = self.summarizer.summarize(
                working["messages"][cursor:cutoff],
                working.get("summary", ""),
            )
            self._record_summary(working, result)
            if not result.summarized_message_count:
                logger.warning("摘要未返回有效纯文本，保留原摘要及消息游标。")
                break
        return self._updates(working)

    async def abefore_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """异步链路只 await 异步摘要模型；与同步链路使用相同状态协议。"""
        working = cast(SearchState, {**state, **extract_search_progress(state)})
        cutoff = self._summary_cutoff(working)
        while self.summarizer is not None and working.get("summarized_message_count", 0) < cutoff:
            cursor = working.get("summarized_message_count", 0)
            result = await self.summarizer.asummarize(
                working["messages"][cursor:cutoff],
                working.get("summary", ""),
            )
            self._record_summary(working, result)
            if not result.summarized_message_count:
                logger.warning("摘要未返回有效纯文本，保留原摘要及消息游标。")
                break
        return self._updates(working)

    def prepare_request(self, request: ModelRequest[None]) -> ModelRequest[None]:
        """在技能与 TODO 注入之后计算预算，在轨迹 middleware 之前裁剪副本。"""
        fixed = estimate_tokens([request.system_message]) if request.system_message else 0
        definitions = [convert_to_openai_tool(tool) for tool in request.tools]
        fixed += estimate_text_tokens(json.dumps(definitions, ensure_ascii=False)) + 32
        budget = self.config.max_tokens - self.config.reserve_output_tokens - fixed
        state = cast(SearchState, {**request.state, "messages": request.messages})
        selected = select_context(state, budget, config=self.config)
        return request.override(
            messages=repair_tool_messages(selected),
            model_settings={
                **request.model_settings,
                "max_tokens": self.config.reserve_output_tokens,
            },
        )

    def after_agent(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """补录最终候选回答及 return_direct 工具结果。"""
        return extract_search_progress(state)

    async def aafter_agent(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        return self.after_agent(state, runtime)

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
