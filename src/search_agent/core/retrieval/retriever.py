"""三类检索与融合的统一入口，业务算法待实现。"""

from pathlib import Path

from .schemas import RetrievalMethod, RetrievalResult


class Retriever:
    """统一协调 BM25、encoder dense、instruction dense 和 RRF。

    TODO：组装三个后端，按 method 路由；hybrid 多路召回后进行 RRF 融合。
    每个后端使用独立索引子目录，不增加 Agent 工具。构造阶段不加载重型模型。
    """

    def __init__(self, index_path: str | Path) -> None:
        """记录索引根目录。TODO：接收三类后端各自的模型配置。"""
        self.index_path = Path(index_path)

    def build_index(self, corpus_path: str | Path, method: RetrievalMethod = "hybrid") -> None:
        """TODO：离线构建选定方法的索引；hybrid 构建三类索引并保存配置。"""
        raise NotImplementedError

    def load_index(self, method: RetrievalMethod = "hybrid") -> None:
        """TODO：只加载所选方法的模型与索引；查询之间共用实例。"""
        raise NotImplementedError

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: RetrievalMethod = "hybrid",
        instruction: str = "Find evidence answering the query.",
    ) -> list[RetrievalResult]:
        """输入搜索词、数量、方法和检索意图，输出统一排序结果。

        TODO：sparse/dense/instruction 路由对应后端；hybrid 多路召回再融合。
        instruction 仅用于指令检索分支。返回原始 doc_id 供 DocumentStore 读取。
        真正完成查询但无命中时才返回 []；未实现必须抛出 NotImplementedError。
        """
        raise NotImplementedError
