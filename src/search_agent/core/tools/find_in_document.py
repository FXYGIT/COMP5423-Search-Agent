from __future__ import annotations

from langchain.tools import tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from search_agent.core.document.schemas import FindResult
from search_agent.core.document.store import get_document_store


class FindInDocumentInput(BaseModel):
    doc_id: str = Field(
        min_length=1,
        description="需要查找的原始文档 ID。",
    )
    text: str = Field(
        min_length=1,
        description="需要查找的字面文本，例如人名、年份或短语。",
    )
    case_sensitive: bool = Field(
        default=False,
        description="是否区分大小写。",
    )
    max_matches: int = Field(
        default=10,
        ge=1,
        le=30,
        description="最多返回的匹配数量。",
    )
    context_chars: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="每个匹配位置前后最多返回的上下文字符数量。",
    )


@tool(args_schema=FindInDocumentInput)
def find_in_document(
    doc_id: str,
    text: str,
    case_sensitive: bool = False,
    max_matches: int = 10,
    context_chars: int = 200,
) -> FindResult:
    """在指定文档中查找字面文本，返回匹配位置及附近原文。

    用于定位人名、日期、机构名称或具体线索。
    不支持语义搜索或正则表达式。
    匹配位置使用原文字符偏移，可传给 open_document 进一步阅读。
    """
    if not text.strip():
        raise ToolException("查找文本不能为空或仅包含空白字符。")

    store = get_document_store()

    try:
        result = store.find(
            doc_id=doc_id,
            text=text,
            case_sensitive=case_sensitive,
            max_matches=max_matches,
            context_chars=context_chars,
        )
    except KeyError:
        raise ToolException(f"文档不存在：{doc_id}。请使用检索结果中的有效文档 ID。") from None

    return result
