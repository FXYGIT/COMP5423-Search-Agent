"""Agent 的原文阅读工具入口。"""

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from search_agent.core.document import DocumentStore


def create_open_document_tool(store: DocumentStore) -> BaseTool:
    """把工具绑定到共享的语料访问层，避免每个工具各自加载整库。"""

    @tool(
        "open_document",
        description=(
            "Read a slice of one fixed-corpus document. doc_id comes from retrieve; "
            "start is a character offset; max_chars defaults to DOCUMENT__MAX_CHARS. "
            "Returns {status, result} where result contains text, source_url, start, end, "
            "total_chars and next_start for continued reading."
        ),
    )
    def open_document(doc_id: str, start: int = 0, max_chars: int | None = None) -> dict[str, Any]:
        """读取指定文档的一段原文，用于核对检索片段和验证候选答案。

        doc_id 来自 retrieve；start 是字符偏移；max_chars 默认使用 DOCUMENT__MAX_CHARS。
        TODO：调用 DocumentStore.read，成功返回 {status: "ok", result: DocumentPage}，
        包含正文、来源 URL、区间和下一读取位置。
        """
        try:
            page = store.read(doc_id, start=start, max_chars=max_chars)
        except KeyError:
            return {"status": "error", "message": f"未知文档 ID：{doc_id}", "result": None}
        except ValueError as error:
            return {"status": "error", "message": str(error), "result": None}
        return {"status": "ok", "result": page.model_dump()}

    return open_document
