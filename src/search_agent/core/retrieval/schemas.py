from pydantic import BaseModel

from search_agent.core.config import RetrievalMethod as RetrievalMethod


class RetrievalResult(BaseModel):
    """统一检索结果：原始文档 ID、相关性分数、正文片段和来源 URL。

    检索器返回按 score 降序排列的列表，工具将其序列化给模型。
    """

    doc_id: str
    score: float
    snippet: str
    source_url: str | None = None
