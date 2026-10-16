"""统一检索入口的骨架。MVP 只规划一种稀疏检索，不预设多后端路由。"""

from pathlib import Path

from .schemas import RetrievalResult


class Retriever:
    """负责建立/读取 BM25 索引，并按查询返回排序后的文档。

    作业最低要求实现一类检索；MVP 选择 BM25。
    Agent 始终只通过 retrieve 工具调用此类。
    稠密检索、指令检索和融合属于后续可选扩展，目前不增加抽象层。
    """

    def __init__(self, index_path: str | Path) -> None:
        """记录本地索引路径。TODO：加载已构建的 BM25 索引。"""
        self.index_path = Path(index_path)

    def build_index(self, corpus_path: str | Path) -> None:
        """从固定语料构建 BM25 索引并保存到 index_path。

        TODO：读取文档，分词，计算词频和文档频率，保存索引与原始 doc_id 映射。
        该函数由离线准备流程调用，不在每次工具检索时重建索引。
        """
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """返回 BM25 分数最高的 top_k 篇不同文档。

        TODO：查询分词、计算相关性、排序，并附上正文片段和来源 URL。
        返回原始文档 ID，确保 open_document 能打开对应原文。
        """
        raise NotImplementedError
