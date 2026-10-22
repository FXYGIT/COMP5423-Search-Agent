"""三类检索与融合的统一入口，业务算法待实现。"""

from pathlib import Path

from search_agent.core.config import AppConfig, get_config

from .schemas import RetrievalMethod, RetrievalResult


class Retriever:
    """统一协调 BM25、encoder dense、instruction dense 和 RRF。

    TODO：组装三个后端，按 method 路由；hybrid 多路召回后进行 RRF 融合。
    每个后端使用独立索引子目录，不增加 Agent 工具。构造阶段不加载重型模型。
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """共用 .env 配置快照，TODO：据此组装后端；构造阶段不加载模型。"""
        self.config = config or get_config()

    def build_index(
        self, corpus_path: str | Path | None = None, method: RetrievalMethod | None = None
    ) -> None:
        """TODO：离线构建选定方法的索引；hybrid 构建三类索引并保存配置。"""
        raise NotImplementedError

    def load_index(self, method: RetrievalMethod | None = None) -> None:
        """TODO：默认 config.retrieval.method；经共享 chroma 模块打开所需模型/集合。

        dense 使用 encoder，instruction 使用 instruction，hybrid 加载两者和 BM25。
        所有路径和模型来自 self.config；查询之间共用实例。
        """
        raise NotImplementedError("TODO：按共享配置加载 BM25/Chroma 检索后端。")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        method: RetrievalMethod | None = None,
        instruction: str | None = None,
    ) -> list[RetrievalResult]:
        """输入搜索词、数量、方法和检索意图，输出统一排序结果。

        TODO：sparse/dense/instruction 路由对应后端；hybrid 多路召回再融合。
        instruction 仅用于指令检索分支。返回原始 doc_id 供 DocumentStore 读取。
        省略参数使用 self.config.retrieval 与 self.config.embedding 中的默认值。
        真正完成查询但无命中时才返回 []；未实现必须抛出 NotImplementedError。
        """
        raise NotImplementedError
