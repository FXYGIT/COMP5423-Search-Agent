"""建库和查询共用的 embedding/Chroma 接口，评估不另建一套参数。"""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from search_agent.core.config import AppConfig, EmbeddingCategory, get_config


class TransformerEmbeddings(Embeddings):
    """把 HuggingFace encoder 包成 LangChain Embeddings 适配器。

    文档和查询共用同一个模型实例；两类模型的差异集中在三处：
    池化方式（encoder 用 CLS，指令模型用最后一个有效 token）、
    查询前缀（指令模型需要 ``Instruct: ...\\nQuery: ...``）、以及模型名。
    """

    def __init__(
        self,
        model_name: str,
        *,
        category: EmbeddingCategory = "encoder",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
        query_instruction: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        if not model_name:
            raise ValueError(f"未配置 {category} embedding 模型")
        self.model_name = model_name
        self.category = category
        self.device = device
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        """首次使用时加载模型；之后复用。"""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if self.device != "cpu":
            self._model.to(self.device)
        self._model.eval()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码；池化方式按类别选择。"""
        import torch

        self._load()
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention = encoded["attention_mask"].to(self.device)
            with torch.inference_mode():
                output = self._model(input_ids=input_ids, attention_mask=attention)
            hidden = output.last_hidden_state
            if self.category == "instruction":
                lengths = attention.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
            else:
                pooled = hidden[:, 0]
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.float().cpu().numpy())
        return np.vstack(vectors).tolist()  # type: ignore[no-any-return]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档侧不加查询前缀。"""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """查询侧：指令模型需要任务前缀，encoder 模型按原样编码。"""
        if self.category == "instruction" and self.query_instruction:
            text = f"Instruct: {self.query_instruction}\nQuery: {text}"
        return self._encode([text])[0]


def create_embeddings(
    config: AppConfig | None = None, *, category: EmbeddingCategory = "encoder"
) -> Embeddings:
    """从共享 embedding 配置选择模型，返回文档和查询共用的编码适配器。

    TODO：加载 config.embedding.model_for(category)，使用同一 device、batch_size、
    normalize 和 query_instruction；遵循模型的 query/document 输入规范。
    不用随机向量或其他模型兜底。
    """
    config = config or get_config()
    model_name = config.embedding.model_for(category)
    if not model_name:
        raise ValueError(
            f"未配置 embedding 模型：请在 .env 设置 "
            f"{'EMBEDDING__DENSE_MODEL' if category == 'encoder' else 'EMBEDDING__INSTRUCTION_MODEL'}。"
        )
    return TransformerEmbeddings(
        model_name,
        category=category,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
        normalize=config.embedding.normalize,
        query_instruction=config.embedding.query_instruction,
        trust_remote_code=config.embedding.trust_remote_code,
    )


def open_chroma(
    embeddings: Embeddings,
    config: AppConfig | None = None,
    *,
    category: EmbeddingCategory = "encoder",
    create: bool = False,
) -> Chroma:
    """共享目录和 collection；建库 create=True，检索/评估只打开已有集合。

    TODO：查询时核对建库 manifest，不能把不同编码模型的向量混入同一集合。
    """
    config = config or get_config()
    path = config.chroma.path
    if not create and not path.is_dir():
        raise FileNotFoundError(f"Chroma 不存在：{path.resolve()}；请先运行建库流程。")
    return Chroma(
        persist_directory=str(path),
        collection_name=config.chroma.collection_for(category),
        embedding_function=embeddings,
        create_collection_if_not_exists=create,
    )
