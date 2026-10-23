"""增量摘要接口；不在骨架阶段调用额外 LLM 或假装完成压缩。"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel

from search_agent.core.trajectory import TrajectoryEntry


class SummaryResult(BaseModel):
    """累计摘要、新覆盖的消息数及本次摘要调用的完整轨迹。

    TODO：记录实际 system/user/assistant 消息及 assistant.usage，追加到主 state.trajectory。
    总量通过 trajectory_totals 统一计算，不能只提供一个孤立的 total_tokens 数字。
    辅助调用不进入主 messages，因此不能增加 trajectory_message_count。
    """

    summary: str
    summarized_message_count: int
    trajectory: list[TrajectoryEntry]


class ConversationSummarizer:
    """将已完成的历史轮次与旧摘要合并，保留事实来源、未决问题和 TODO。

    TODO：达到 token 阈值时调用作业指定主模型；不能切断 AI/tool 成对消息。
    摘要不覆盖原始 messages，只作为模型请求视图的一部分。
    """

    def summarize(self, messages: list[AnyMessage], previous_summary: str = "") -> SummaryResult:
        """输入待压缩的完整旧轮次和累计摘要，输出新累计摘要及实际 API 用量。

        TODO：保留 doc_id 引用与未完成任务；不能把候选答案写成已证实事实。
        """
        raise NotImplementedError
