"""工具消息配对的最小实现，用于中断后恢复和将来的上下文裁剪。"""

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage


def repair_tool_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """返回模型可消费的消息视图，保留完整工具对，移除悬空调用与孤立结果。

    仅匹配 AI 消息后连续的 ToolMessage，不能用后续轮次的同名工具结果补位。
    多工具调用只保留已返回的部分，并按调用顺序排列。没有结果的调用不重试，
    不伪造成功/失败 ToolMessage；原始输入和轨迹不变。
    """
    result: list[AnyMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if isinstance(message, ToolMessage):
            continue
        if not isinstance(message, AIMessage) or not message.tool_calls:
            result.append(message)
            continue
        replies: dict[str, ToolMessage] = {}
        while index < len(messages) and isinstance(messages[index], ToolMessage):
            reply = messages[index]
            if isinstance(reply, ToolMessage):
                replies[reply.tool_call_id] = reply
            index += 1
        calls = [call for call in message.tool_calls if call["id"] in replies]
        content = message.content
        if isinstance(content, list):
            ids = {call["id"] for call in calls}
            content = [
                block
                for block in content
                if not isinstance(block, dict)
                or block.get("type") not in {"tool_call", "tool_use"}
                or block.get("id") in ids
            ]
        if calls or content:
            # OpenAI 适配器在 tool_calls 为空时可能回读原始 provider 字段；一起移除。
            extras = {
                key: value
                for key, value in message.additional_kwargs.items()
                if key != "tool_calls"
            }
            result.append(
                message.model_copy(
                    update={
                        "tool_calls": calls,
                        "content": content,
                        "additional_kwargs": extras,
                    }
                )
            )
            result.extend(replies[call["id"]] for call in calls if call["id"] is not None)
    return result
