"""Instruction-aware dense retrieval 骨架。"""

from pathlib import Path

from .schemas import RetrievalResult


class InstructionRetriever:
    """使用 Qwen3-Embedding/E5-Mistral 等 LLM-based embedding 模型。

    instruction 描述检索意图，与 query 共同用于查询编码；文档编码遵循模型规范。
    TODO：独立索引和模型配置；不能直接复用其他模型产生的向量。
    """

    def __init__(self, index_path: str | Path, model_name: str) -> None:
        """记录指令嵌入模型和索引目录，保持构造阶段无 IO。"""
        self.index_path = Path(index_path)
        self.model_name = model_name

    def build_index(self, corpus_path: str | Path) -> None:
        """TODO：按模型要求编码语料，保存索引、块偏移和原始 doc_id。"""
        raise NotImplementedError

    def load_index(self) -> None:
        """TODO：读取该模型专用的索引及元信息。"""
        raise NotImplementedError

    def retrieve(
        self, query: str, top_k: int = 5, instruction: str = "Find evidence answering the query."
    ) -> list[RetrievalResult]:
        """TODO：使用 instruction + query 编码，召回并归并成排序后的文档列表。"""
        raise NotImplementedError
