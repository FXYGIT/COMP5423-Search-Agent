"""Agent 的原文阅读工具入口。"""

from typing import Any

from langchain.tools import tool


@tool
def open_document(doc_id: str, start: int = 0, max_chars: int = 4000) -> dict[str, Any]:
    """读取指定文档的一段原文，用于核对检索片段和验证候选答案。

    doc_id 来自 retrieve；start 是字符偏移；max_chars 是读取长度。
    TODO：调用 DocumentStore.read，成功返回 {status: "ok", result: DocumentPage}，
    包含正文、来源 URL、区间和下一读取位置。
    """
    return {"status": "not_implemented", "message": "文档阅读尚未实现。", "result": None}
