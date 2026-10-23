"""Agent 与评估共用的消息协议，对齐根目录 test_prediction.jsonl 示例。

仅转换真实消息与 API usage；不估算 token，不执行检索或摘要业务。
LangChain 内部消息保持原样，提交轨迹使用 OpenAI 的 role/function 格式。
"""

import json
from collections.abc import Sequence
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field


class TraceModel(BaseModel):
    """轨迹字段采用显式协议，拒绝遗留的 type/text 自定义事件。"""

    model_config = ConfigDict(extra="forbid")


class TokenUsage(TraceModel):
    """单次 API 实际用量；total_tokens 直接取服务端返回值。"""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ToolFunction(TraceModel):
    """工具名称和 JSON 编码的参数字符串，arguments 不能直接写成字典。"""

    name: str
    arguments: str


class ToolCall(TraceModel):
    """一个模型发起的调用；并行发出 N 个调用时计为 N 次。"""

    id: str
    type: Literal["function"] = "function"
    function: ToolFunction


class SystemEntry(TraceModel):
    """实际传入模型的系统提示词，包括 middleware 注入的内容。"""

    role: Literal["system"] = "system"
    content: str


class UserEntry(TraceModel):
    """用户问题；不能把评估参考答案写入该消息。"""

    role: Literal["user"] = "user"
    content: str


class AssistantEntry(TraceModel):
    """完整模型响应；仅工具调用时 content=null，缺失 usage 时禁止提交。

    usage 可暂缺以支持无用量信息的聊天；不会序列化为空 usage 或虚假的 0。
    没有调用时省略 tool_calls，与示例最终回答一致。
    """

    role: Literal["assistant"] = "assistant"
    content: str | None
    tool_calls: list[ToolCall] | None = Field(default=None, exclude_if=lambda value: value is None)
    usage: TokenUsage | None = Field(default=None, exclude_if=lambda value: value is None)


class ToolEntry(TraceModel):
    """完整工具返回内容，以 tool_call_id 与对应调用关联，不截断 CLI 预览之外的数据。"""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    name: str
    content: str


TrajectoryEntry = Annotated[
    SystemEntry | UserEntry | AssistantEntry | ToolEntry, Field(discriminator="role")
]


def message_to_entry(message: BaseMessage) -> TrajectoryEntry:
    """把一条 LangChain 消息转换为提交条目，保留工具结果与 API 计数。

    当前文本 Agent 只使用 system/human/ai/tool。API 用量优先读取 LangChain
    usage_metadata；兼容原始 response_metadata.token_usage。工具块内容若为列表，
    用 JSON 字符串保留全部内容；不把工具输出解析成新的提交字段。
    """
    if isinstance(message, SystemMessage):
        return SystemEntry(content=message.text)
    if isinstance(message, HumanMessage):
        return UserEntry(content=message.text)
    if isinstance(message, ToolMessage):
        return ToolEntry(
            tool_call_id=message.tool_call_id,
            name=message.name or "",
            content=message.content
            if isinstance(message.content, str)
            else json.dumps(message.content, ensure_ascii=False),
        )
    if isinstance(message, AIMessage):
        usage = None
        if message.usage_metadata is not None:
            values = message.usage_metadata
            usage = TokenUsage(
                prompt_tokens=values["input_tokens"],
                completion_tokens=values["output_tokens"],
                total_tokens=values["total_tokens"],
            )
        elif raw := message.response_metadata.get("token_usage"):
            usage = TokenUsage(**{key: raw[key] for key in TokenUsage.model_fields})
        calls = [
            ToolCall(
                id=call["id"],
                function=ToolFunction(
                    name=call["name"], arguments=json.dumps(call["args"], ensure_ascii=False)
                ),
            )
            for call in message.tool_calls
        ]
        return AssistantEntry(content=message.text or None, tool_calls=calls or None, usage=usage)
    raise ValueError(f"不支持的轨迹消息类型：{message.type}")


def trajectory_totals(entries: Sequence[TrajectoryEntry]) -> tuple[int, int | None]:
    """返回发起的工具次数与 API token 总量；任何模型缺 usage 时总量为 None。"""
    responses = [entry for entry in entries if isinstance(entry, AssistantEntry)]
    calls = sum(len(entry.tool_calls or []) for entry in responses)
    if any(entry.usage is None for entry in responses):
        return calls, None
    return calls, sum(entry.usage.total_tokens for entry in responses if entry.usage is not None)
