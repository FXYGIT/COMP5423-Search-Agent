"""Agent 的文档内查找工具入口。"""

from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from search_agent.core.document import DocumentStore


def create_find_in_document_tool(store: DocumentStore) -> BaseTool:
    """把工具绑定到共享的语料访问层，查找只在同一固定语料内进行。"""

    @tool(
        "find_in_document",
        description=(
            "Locate a literal name, date or phrase inside one fixed-corpus document. "
            "doc_id is the original document id, text is the literal string to find. "
            "Returns {status, result} where result lists matches with start/end offsets "
            "and surrounding context that can be passed to open_document."
        ),
    )
    def find_in_document(doc_id: str, text: str) -> dict[str, Any]:
        """在指定原文中定位人名、日期或短语，返回偏移和附近上下文。

        doc_id 是原始文档 ID，text 是字面查找词句。
        TODO：调用 DocumentStore.find，成功返回 {status: "ok", result: FindResult}，
        提供原文偏移以供 open_document 继续阅读。
        """
        try:
            found = store.find(doc_id, text)
        except KeyError:
            return {"status": "error", "message": f"未知文档 ID：{doc_id}", "result": None}
        except ValueError as error:
            return {"status": "error", "message": str(error), "result": None}
        return {"status": "ok", "result": found.model_dump()}

    return find_in_document
