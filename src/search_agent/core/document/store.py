"""固定语料访问层的骨架，提供文档读取能力，不负责搜索排序。"""

from collections.abc import Iterator
from pathlib import Path
from threading import Lock
from typing import Any

from datasets import load_from_disk  # type: ignore[import-untyped]

from search_agent.core.config import DocumentConfig, get_config

from .schemas import Document, DocumentPage, FindResult, TextMatch


class DocumentStore:
    """加载 BrowseComp-Plus 本地语料，按原始 doc_id 访问文档。

    后续在此建立 doc_id 到原文的映射；检索器和文档工具共用该访问层。
    默认数据、路径和参数合法。阅读优化和查找回退由 DocumentReader 负责。
    """

    def __init__(
        self,
        corpus_path: str | Path | None = None,
        *,
        config: DocumentConfig | None = None,
    ) -> None:
        """构造时只保存配置，首次读取才打开语料并建立文档 ID 映射。"""
        self.config = config if config is not None else get_config().document
        self.corpus_path = Path(corpus_path) if corpus_path is not None else self.config.corpus_path
        self._corpus: tuple[Any, dict[str, int]] | None = None
        self._load_lock = Lock()

    def _load_corpus(self) -> tuple[Any, dict[str, int]]:
        with self._load_lock:
            if self._corpus is None:
                dataset = load_from_disk(str(self.corpus_path))
                id_to_index = {str(doc_id): index for index, doc_id in enumerate(dataset["docid"])}
                self._corpus = dataset, id_to_index
            return self._corpus

    def get(self, doc_id: str) -> Document:
        """根据原始文档 ID 返回全文及来源 URL，供工具读取和证据引用。"""
        dataset, id_to_index = self._load_corpus()
        row = dataset[id_to_index[doc_id]]
        return Document(doc_id=str(row["docid"]), text=row["text"], source_url=row["url"])

    def iter_documents(self) -> Iterator[Document]:
        """逐篇提供原文，供检索索引构建使用，不在此进行分词或打分。"""
        dataset, _ = self._load_corpus()
        for row in dataset:
            yield Document(doc_id=str(row["docid"]), text=row["text"], source_url=row["url"])

    def read(self, doc_id: str, start: int = 0, max_chars: int | None = None) -> DocumentPage:
        """读取从 start 开始的原文片段。

        TODO：通过 get 获取原文，切出 max_chars 个字符，返回区间和 next_start。
        使用原文字符偏移，便于 find 的定位结果直接交给阅读工具。
        未提供 max_chars 时读取 self.config.max_chars。
        """
        if start < 0:
            raise ValueError("start must be non-negative")
        document = self.get(doc_id)
        if max_chars is None:
            max_chars = self.config.max_chars
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        total_chars = len(document.text)
        start = min(start, total_chars)
        end = min(start + max_chars, total_chars)
        text = document.text[start:end]
        if end < total_chars:
            next_start = end
        else:
            next_start = None
        return DocumentPage(
            doc_id=document.doc_id,
            text=text,
            source_url=document.source_url,
            start=start,
            end=end,
            total_chars=total_chars,
            next_start=next_start,
        )

    def find(self, doc_id: str, text: str) -> FindResult:
        """在原文中查找字面词句，返回匹配偏移及附近上下文。

        TODO：通过 get 获取原文，定位 text 并组装 TextMatch 列表。
        这是辅助阅读工具，不承担语义检索。
        """
        if text == "":
            raise ValueError("search text must not be empty")
        document = self.get(doc_id)
        content = document.text
        content_char = 100
        matches: list[TextMatch] = []
        start = content.find(text)
        while start != -1:
            end = start + len(text)
            context_start = max(0, start - content_char)
            context_end = min(len(content), end + content_char)
            matches.append(
                TextMatch(
                    start=start,
                    end=end,
                    matched_text=content[start:end],
                    context_start=context_start,
                    context_end=context_end,
                    context=content[context_start:context_end],
                )
            )

            start = content.find(text, end)
        return FindResult(
            doc_id=doc_id,
            text=text,
            matches=matches,
        )
