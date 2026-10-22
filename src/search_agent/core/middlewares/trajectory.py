"""作业第 4 节：机器可读轨迹、工具调用次数和 API token 用量的实现位置。"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from search_agent.core.state import SearchState


class TrajectoryMiddleware(AgentMiddleware[SearchState, None]):
    """为未来批量评估整理每道题的过程记录，当前注册但仅透传。

    轨迹包含模型输出、工具名称、输入和结果；不能人工编造。
    total_tokens 应使用 API 返回的 usage，而不是估算字符数。
    CLI 的实时展示直接使用 LangChain 流，不依赖此统计模块。
    """

    state_schema = SearchState

    def after_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """TODO：累加本次模型响应的 usage.total_tokens 和发起的工具调用数量。"""
        return {}

    def after_agent(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """TODO：按顺序整理 messages，返回 trajectory、tool_calls、total_tokens。

        批量评估入口再加上 id、question、answer，写入提交 JSONL。
        当前多轮 CLI 只演示交互，不生成评估文件。
        """
        return {}

    async def aafter_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """异步统计入口，TODO 内容同 after_model。"""
        return self.after_model(state, runtime)

    async def aafter_agent(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """异步收尾入口，TODO 内容同 after_agent；不得把未统计显示成 0 token。"""
        return self.after_agent(state, runtime)
