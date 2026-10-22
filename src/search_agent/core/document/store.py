"""固定语料访问层的骨架，提供文档读取能力，不负责搜索排序。"""

from collections.abc import Iterator
from pathlib import Path

from .schemas import Document, DocumentPage, FindResult


class DocumentStore:
    """加载 BrowseComp-Plus 本地语料，按原始 doc_id 访问文档。

    后续在此建立 doc_id 到原文的映射；检索器和文档工具共用该访问层。
    默认数据、路径和参数合法。阅读优化和查找回退由 DocumentReader 负责。
    """

    def __init__(self, corpus_path: str | Path) -> None:
        """记录语料路径。TODO：加载本地数据集并建立文档 ID 映射。"""
        self.corpus_path = Path(corpus_path)

    def get(self, doc_id: str) -> Document:
        """根据原始文档 ID 返回全文及来源 URL，供工具读取和证据引用。"""
        raise NotImplementedError

    def iter_documents(self) -> Iterator[Document]:
        """逐篇提供原文，供检索索引构建使用，不在此进行分词或打分。"""
        raise NotImplementedError

    def read(self, doc_id: str, start: int = 0, max_chars: int = 4000) -> DocumentPage:
        """读取从 start 开始的原文片段。

        TODO：通过 get 获取原文，切出 max_chars 个字符，返回区间和 next_start。
        使用原文字符偏移，便于 find 的定位结果直接交给阅读工具。
        """
        raise NotImplementedError

    def find(self, doc_id: str, text: str) -> FindResult:
        """在原文中查找字面词句，返回匹配偏移及附近上下文。

        TODO：通过 get 获取原文，定位 text 并组装 TextMatch 列表。
        这是辅助阅读工具，不承担语义检索。
        """
        raise NotImplementedError
