"""Encoder-based dense retrieval 骨架。"""

from pathlib import Path

from search_agent.core.config import AppConfig, get_config

from .schemas import RetrievalResult


class DenseRetriever:
    """使用 E5/BGE/GTE 等 encoder 编码查询和文档，进行向量检索。

    TODO：模型只初始化一次；明确 query/document 编码方式、归一化与相似度。
    和 instruction retriever 分开实现，避免把同一模型换提示词算作两类方法。
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """和建库共用模型、路径、collection，不下载模型或加载索引。"""
        self.config = config or get_config()
        self.index_path = self.config.chroma.path
        self.model_name = self.config.embedding.dense_model
        self.collection_name = self.config.chroma.dense_collection

    def build_index(self, corpus_path: str | Path) -> None:
        """TODO：切块并批量编码固定语料，保存向量、原始 doc_id/偏移及模型配置。"""
        raise NotImplementedError

    def load_index(self) -> None:
        """TODO：加载向量库及 ID 映射，保证查询使用建库时的同一个模型。"""
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """TODO：复用 chroma.create_embeddings/open_chroma；数量默认 config.retrieval.top_k。

        编码查询、向量召回、块级结果按文档归并，返回 top_k 篇原文。
        """
        raise NotImplementedError
