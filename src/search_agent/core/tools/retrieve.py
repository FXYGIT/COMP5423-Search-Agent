"""唯一检索工具：默认参数来自统一配置，返回结果不伪装成已完成的查询。"""

import logging
from typing import Any

from langchain.tools import tool
from langchain_core.tools import BaseTool

from search_agent.core.config import AppConfig, RetrievalMethod, get_config
from search_agent.core.retrieval import Retriever

logger = logging.getLogger(__name__)


def create_retrieve_tool(config: AppConfig | None = None) -> BaseTool:
    """绑定 Agent 的配置快照；独立使用时在调用工具时读取 .env，不在导入时读取。"""

    @tool("retrieve")
    def retrieve(
        query: str,
        top_k: int | None = None,
        method: RetrievalMethod | None = None,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        """搜索固定文档集合，返回 doc_id、score、snippet 和 source_url。

        可选参数省略时使用 .env：RETRIEVAL__TOP_K / METHOD 和 EMBEDDING__QUERY_INSTRUCTION。
        method 支持 sparse/dense/instruction/hybrid。TODO：底层索引和算法仍待实现。
        """
        settings = config or get_config()
        selected_method = method if method is not None else settings.retrieval.method
        count = top_k if top_k is not None else settings.retrieval.top_k
        intent = instruction if instruction is not None else settings.embedding.query_instruction
        logger.info("检索请求 method=%s top_k=%d", selected_method, count)
        # TODO：后端实现后在工厂生命周期内复用已加载实例，而不是每次查询重载模型。
        backend = Retriever(settings)
        try:
            results = backend.retrieve(query, count, selected_method, intent)
        except NotImplementedError:
            logger.info("检索后端尚未实现")
            return {
                "status": "not_implemented",
                "message": "固定语料检索尚未实现。",
                "result": None,
            }
        return {"status": "ok", "result": [item.model_dump() for item in results]}

    return retrieve


retrieve = create_retrieve_tool()
