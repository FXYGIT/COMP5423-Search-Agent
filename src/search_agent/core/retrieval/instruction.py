"""Instruction-aware dense retrieval 骨架。"""

from pathlib import Path

from search_agent.core.config import AppConfig, get_config

from .schemas import RetrievalResult


class InstructionRetriever:
    """使用 Qwen3-Embedding/E5-Mistral 等 LLM-based embedding 模型。

    instruction 描述检索意图，与 query 共同用于查询编码；文档编码遵循模型规范。
    TODO：独立索引和模型配置；不能直接复用其他模型产生的向量。
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """使用共享指令模型与专属 collection，构造时不加载模型或索引。"""
        self.config = config or get_config()
        self.index_path = self.config.chroma.path
        self.model_name = self.config.embedding.instruction_model
        self.collection_name = self.config.chroma.instruction_collection

    def build_index(self, corpus_path: str | Path) -> None:
        """TODO：按模型要求编码语料，保存索引、块偏移和原始 doc_id。"""
        raise NotImplementedError

    def load_index(self) -> None:
        """TODO：读取该模型专用的索引及元信息。"""
        raise NotImplementedError

    def retrieve(
        self, query: str, top_k: int | None = None, instruction: str | None = None
    ) -> list[RetrievalResult]:
        """TODO：使用 instruction + query 编码，返回按原始文档归并的排名。

        默认使用 config.retrieval.top_k / config.embedding.query_instruction；
        经共享 chroma 模块以 category=instruction 加载模型和集合。
        """
        raise NotImplementedError
