"""稀疏检索骨架：词项匹配，不加载 embedding 模型。"""

from pathlib import Path

from search_agent.core.config import AppConfig, get_config

from .schemas import RetrievalResult


class BM25Retriever:
    """维护 BM25 索引和原始 doc_id 映射，供统一 Retriever 调用。"""

    def __init__(self, config: AppConfig | None = None) -> None:
        """读取统一配置中的稀疏索引目录；构造阶段不读语料。"""
        self.config = config or get_config()
        self.index_path = self.config.retrieval.sparse_index_path

    def build_index(self, corpus_path: str | Path) -> None:
        """输入固定语料目录，输出磁盘索引，无返回值。

        TODO：分词、建立 BM25 统计量，保存索引、分词配置和原始 ID 映射。
        """
        raise NotImplementedError

    def load_index(self) -> None:
        """TODO：从 index_path 恢复索引到当前实例；查询时复用，不重复建库。"""
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """TODO：查询分词、BM25 打分，数量默认 self.config.retrieval.top_k。"""
        raise NotImplementedError
