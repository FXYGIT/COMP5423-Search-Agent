"""无需联网或下载 tokenizer 的预算估算；绝不用于评估 API 用量。"""

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from .messages import repair_tool_messages

TRUNCATED = "\n[Context preview truncated; full result remains in original history.]"


def estimate_text_tokens(text: str) -> int:
    """按 UTF-8 每 3 字节一个 token 估算，中英文混合也计入；不是模型 tokenizer。"""
    return (len(text.encode("utf-8")) + 2) // 3


def estimate_tokens(messages: Sequence[AnyMessage]) -> int:
    """计入角色、调用 ID 和工具参数等开销，避免只计算可见文本。"""
    total = 0
    for message in messages:
        data: dict[str, Any] = {"role": message.type, "content": message.content}
        if isinstance(message, AIMessage) and message.tool_calls:
            data["tool_calls"] = message.tool_calls
        if isinstance(message, ToolMessage):
            data.update(tool_call_id=message.tool_call_id, name=message.name or "")
        total += 8 + estimate_text_tokens(json.dumps(data, ensure_ascii=False))
    return total


def truncate_text(text: str, max_tokens: int, suffix: str = TRUNCATED) -> str:
    """二分截取字符前缀，不切开 Unicode 字符；不足容纳标记时返回空文本。"""
    if estimate_text_tokens(text) <= max_tokens:
        return text
    if estimate_text_tokens(suffix) > max_tokens:
        return ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(text[:middle] + suffix) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix


def message_blocks(messages: list[AnyMessage]) -> list[list[AnyMessage]]:
    """将 AI 和紧接的工具结果组成不可拆开的块，先修复中断后的消息。"""
    blocks: list[list[AnyMessage]] = []
    for message in repair_tool_messages(messages):
        if isinstance(message, ToolMessage):
            blocks[-1].append(message)
        else:
            blocks.append([message])
    return blocks


def fit_block(block: list[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """超长块仅裁剪内容副本；保留真实调用参数、调用 ID 和全部对应工具结果。"""
    result = list(block)
    while estimate_tokens(result) > max_tokens:
        candidates = [
            i
            for i, message in enumerate(result)
            if not isinstance(message, HumanMessage) and message.content
        ]
        if not candidates:
            raise ValueError("上下文预算不足以容纳最新问题或完整工具调用参数；请增加 MAX_TOKENS。")
        index = max(candidates, key=lambda i: estimate_text_tokens(result[i].text))
        message = result[index]
        # JSON 中的换行、引号和反斜杠也有开销，按完整序列化请求二分，
        # 不能从可见正文长度简单减去超额数，否则可能把正文一次清空。
        trial = list(result)
        trial[index] = message.model_copy(update={"content": TRUNCATED})
        if estimate_tokens(trial) > max_tokens:
            result[index] = message.model_copy(update={"content": ""})
            continue
        low, high = 0, len(message.text)
        while low < high:
            middle = (low + high + 1) // 2
            trial[index] = message.model_copy(update={"content": message.text[:middle] + TRUNCATED})
            if estimate_tokens(trial) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        result[index] = message.model_copy(update={"content": message.text[:low] + TRUNCATED})
    return result
