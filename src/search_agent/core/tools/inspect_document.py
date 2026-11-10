"""查询相关的原文片段选择和定位回退入口。"""

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from search_agent.core.document import DocumentReader


def create_inspect_document_tool(reader: DocumentReader) -> BaseTool:
    """把工具绑定到共享的文档阅读器；不发起全库检索，也不抓取外网。"""

    @tool(
        "inspect_document",
        description=(
            "Select query-relevant evidence inside one known document, falling back from "
            "exact matching to normalized matching, passage ranking and sequential reading. "
            "Returns {status, result} with passages (real original offsets), the locating "
            "method, and next_start for further reading."
        ),
    )
    def inspect_document(doc_id: str, query: str, max_chars: int | None = None) -> dict[str, Any]:
        """在指定文档内选择与 query 有关的证据；精确定位失败时逐级回退。

        TODO：调用 DocumentReader.inspect。成功返回 {status: "ok", result: InspectionResult}，
        包含原文片段、偏移、定位方式和续读位置；不在全库检索，不抓取外网。
        max_chars 默认使用 DOCUMENT__MAX_CHARS。
        """
        try:
            inspection = reader.inspect(doc_id, query, max_chars=max_chars)
        except KeyError:
            return {"status": "error", "message": f"未知文档 ID：{doc_id}", "result": None}
        except ValueError as error:
            return {"status": "error", "message": str(error), "result": None}
        return {"status": "ok", "result": inspection.model_dump()}

    return inspect_document
