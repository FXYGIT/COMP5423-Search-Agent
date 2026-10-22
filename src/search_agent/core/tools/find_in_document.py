"""Agent 的文档内查找工具入口。"""

from typing import Any

from langchain.tools import tool


@tool
def find_in_document(doc_id: str, text: str) -> dict[str, Any]:
    """在指定原文中定位人名、日期或短语，返回偏移和附近上下文。

    doc_id 是原始文档 ID，text 是字面查找词句。
    TODO：调用 DocumentStore.find，成功返回 {status: "ok", result: FindResult}，
    提供原文偏移以供 open_document 继续阅读。
    """
    return {"status": "not_implemented", "message": "文档内查找尚未实现。", "result": None}
