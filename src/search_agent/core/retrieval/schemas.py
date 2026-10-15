from __future__ import annotations

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    doc_id: str
    score: float
    snippet: str
    source_url: str | None = None
