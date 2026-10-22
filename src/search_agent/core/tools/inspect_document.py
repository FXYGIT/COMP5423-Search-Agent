"""查询相关的原文片段选择和定位回退入口。"""

from typing import Any

from langchain.tools import tool


@tool
def inspect_document(doc_id: str, query: str, max_chars: int = 4000) -> dict[str, Any]:
    """在指定文档内选择与 query 有关的证据；精确定位失败时逐级回退。

    TODO：调用 DocumentReader.inspect。成功返回 {status: "ok", result: InspectionResult}，
    包含原文片段、偏移、定位方式和续读位置；不在全库检索，不抓取外网。
    """
    return {"status": "not_implemented", "message": "文档定位与回退尚未实现。", "result": None}
