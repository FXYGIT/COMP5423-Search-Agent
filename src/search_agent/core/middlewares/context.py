"""作业 1.2 Feature B：结构化搜索状态及模型上下文控制的实现位置。"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from search_agent.core.state import SearchState


class SearchContextMiddleware(AgentMiddleware[SearchState, None]):
    """整理工具结果，为下一轮模型调用提供搜索进度与选中证据。

    TODO：记录已访问文档、证据和当前子查询；去重或裁剪传给模型的上下文。
    原始 messages 留给轨迹统计，不直接删除原始工具结果。
    本类仅为骨架，当前 agent 不注册；MVP 对话由 CLI 传递完整历史。
    """

    state_schema = SearchState

    def before_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """在下一次推理前，读取新增工具结果并返回 SearchState 的更新字段。

        TODO：提取 doc_id/证据，更新 active_subqueries/candidate_answer。
        后续增加 wrap_model_call，在请求副本中选择上下文，不改原始轨迹。
        """
        raise NotImplementedError
