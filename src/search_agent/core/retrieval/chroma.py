"""建库和查询共用的 embedding/Chroma 接口，评估不另建一套参数。"""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from search_agent.core.config import AppConfig, EmbeddingCategory, get_config


def create_embeddings(
    config: AppConfig | None = None, *, category: EmbeddingCategory = "encoder"
) -> Embeddings:
    """从共享 embedding 配置选择模型，返回文档和查询共用的编码适配器。

    TODO：加载 config.embedding.model_for(category)，使用同一 device、batch_size、
    normalize 和 query_instruction；遵循模型的 query/document 输入规范。
    不用随机向量或其他模型兜底。
    """
    config = config or get_config()
    raise NotImplementedError(
        f"TODO：实现 {category} embedding 适配器（模型：{config.embedding.model_for(category) or '未配置'}）。"
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
