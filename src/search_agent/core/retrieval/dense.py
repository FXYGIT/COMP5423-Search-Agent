"""Encoder-based dense retrieval 骨架。"""

from pathlib import Path

from .schemas import RetrievalResult


class DenseRetriever:
    """使用 E5/BGE/GTE 等 encoder 编码查询和文档，进行向量检索。

    TODO：模型只初始化一次；明确 query/document 编码方式、归一化与相似度。
    和 instruction retriever 分开实现，避免把同一模型换提示词算作两类方法。
    """

    def __init__(self, index_path: str | Path, model_name: str) -> None:
        """记录索引目录和 encoder 模型名，不下载模型、不加载索引。"""
        self.index_path = Path(index_path)
        self.model_name = model_name

    def build_index(self, corpus_path: str | Path) -> None:
        """TODO：切块并批量编码固定语料，保存向量、原始 doc_id/偏移及模型配置。"""
        raise NotImplementedError

    def load_index(self) -> None:
        """TODO：加载向量库及 ID 映射，保证查询使用建库时的同一个模型。"""
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """TODO：编码查询、向量召回、块级结果按文档归并，返回 top_k 篇原文。"""
        raise NotImplementedError
