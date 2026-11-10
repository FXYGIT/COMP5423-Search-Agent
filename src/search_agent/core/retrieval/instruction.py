"""Instruction-aware dense retrieval 骨架。"""

from __future__ import annotations

import numpy as np

from search_agent.core.config import AppConfig, get_config

from .dense import DenseRetriever
from .schemas import RetrievalResult


class InstructionRetriever(DenseRetriever):
    """使用 Qwen3-Embedding/E5-Mistral 等 LLM-based embedding 模型。

    instruction 描述检索意图，与 query 共同用于查询编码；文档编码遵循模型规范。
    TODO：独立索引和模型配置；不能直接复用其他模型产生的向量。

    实现说明：复用 DenseRetriever 的建库/加载/归并逻辑，只覆盖两处差异：
    查询侧加 ``Instruct: ...\\nQuery: ...`` 前缀，池化改用最后一个有效 token。
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """使用共享指令模型与专属 collection，构造时不加载模型或索引。"""
        super().__init__(config or get_config())
        self.model_name = self.config.embedding.instruction_model
        self.collection_name = self.config.chroma.instruction_collection
        # 指令模型上下文更长，段落放大到 512 词以减少向量数量。
        self.passage_words = 512

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Qwen3-Embedding 使用最后一个有效 token 的隐状态作为句向量。"""
        import torch

        self._ensure_model()
        batch_size = max(1, self.config.embedding.batch_size)
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=int(self.config.embedding.max_length),
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self._device)
            attention = encoded["attention_mask"].to(self._device)
            with torch.inference_mode():
                output = self._model(input_ids=input_ids, attention_mask=attention)
            hidden = output.last_hidden_state
            lengths = attention.sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            if self.config.embedding.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.float().cpu().numpy())
        return np.vstack(vectors)

    def _query_vector(self, query: str, instruction: str | None = None) -> np.ndarray:
        """指令模型要求查询带 ``Instruct: ...\\nQuery: ...`` 前缀。"""
        task = instruction or self.config.embedding.query_instruction
        return self._encode([f"Instruct: {task}\nQuery: {query}"])[0]  # type: ignore[no-any-return]

    def retrieve(
        self, query: str, top_k: int | None = None, instruction: str | None = None
    ) -> list[RetrievalResult]:
        """TODO：使用 instruction + query 编码，返回按原始文档归并的排名。

        默认使用 config.retrieval.top_k / config.embedding.query_instruction；
        经共享 chroma 模块以 category=instruction 加载模型和集合。
        """
        return super().retrieve(query, top_k, instruction)
