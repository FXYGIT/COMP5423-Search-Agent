"""验证/测试数据、提交文件及本地报告的输入输出协议。"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from search_agent.core.trajectory import TrajectoryEntry, trajectory_totals


class EvaluationQuestion(BaseModel):
    """本地数据行；answer 是评分标签，不能传入 Agent 的消息或工具。"""

    id: str
    question: str
    answer: str | None = None


class PredictionRecord(BaseModel):
    """作业要求的六字段记录，禁止附加参考答案等其他字段。

    trajectory 为程序实际运行的完整过程，total_tokens 包含所有 LLM 调用的 API 用量。
    运行失败或用量缺失时不应构造本对象；不能用空轨迹或 0 token 替代缺失统计。
    """

    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    answer: str
    trajectory: list[TrajectoryEntry] = Field(min_length=1)
    tool_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        """提交总数必须来自逐条轨迹，缺少 usage 时不能用部分用量冒充总量。"""
        calls, tokens = trajectory_totals(self.trajectory)
        if tokens is None:
            raise ValueError("trajectory 中的模型响应缺少 API usage。")
        if self.tool_calls != calls:
            raise ValueError("tool_calls 必须等于 trajectory 中发起的工具调用数量。")
        if self.total_tokens != tokens:
            raise ValueError("total_tokens 必须等于 trajectory 中所有 usage.total_tokens 之和。")
        return self


class QuestionResult(BaseModel):
    """报告中的逐题对照；只用于本地分析，不写入提交 JSONL。"""

    id: str
    prediction: str
    reference: str | None
    exact_match: bool | None
    tool_calls: int
    total_tokens: int


class EvaluationReport(BaseModel):
    """EM 为 [0, 1] 比例；无参考答案时为 None，Markdown 中展示百分比。

    报告只说明当前数据上的本地指标，不表示满足全部作业功能或获得班级排名分数。
    """

    question_count: int
    labelled_count: int
    normalized_em: float | None
    total_tool_calls: int
    total_tokens: int
    mean_tool_calls: float
    mean_tokens: float
    results: list[QuestionResult]
