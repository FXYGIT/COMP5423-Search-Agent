"""评估工具组装接口；模型和 Chroma 配置统一由 core 管理。"""

from langchain_core.tools import BaseTool

from search_agent.core.config import AppConfig
from search_agent.core.retrieval import Retriever


def create_evaluation_tools(retriever: Retriever, config: AppConfig) -> list[BaseTool]:
    """输入已加载的统一检索器和同一配置快照，输出评估 Agent 的业务工具。

    TODO：将唯一 retrieve 绑定到 retriever；普通/指令向量模型及 collection
    由 core.retrieval.chroma 依据共享配置打开，hybrid 在内部融合多个后端。
    原文工具使用 config.document.corpus_path，技能使用 config.skills。
    不另外定义 embedding、collection 或语料路径，不退回未连接索引的占位工具。
    """
    raise NotImplementedError("TODO：将统一检索器、固定原文和技能绑定到真实 Agent 工具。")
