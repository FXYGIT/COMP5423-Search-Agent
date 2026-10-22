"""三类检索、融合及统一入口的接口骨架。"""

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from .instruction import InstructionRetriever
from .retriever import Retriever
from .schemas import RetrievalMethod, RetrievalResult

__all__ = [
    "Retriever",
    "RetrievalMethod",
    "RetrievalResult",
    "BM25Retriever",
    "DenseRetriever",
    "InstructionRetriever",
    "reciprocal_rank_fusion",
]
