from collections.abc import Iterator
from pathlib import Path

from search_agent.core.document import Document, DocumentPage, FindResult


class DocumentStore:
    """固定文档集合的访问入口。

    职责：
        1. 加载本地 BrowseComp-Plus 文档集合。
        2. 根据文档 ID 获取完整原文。
        3. 支持分段阅读和文档内查找。
        4. 为检索索引构建提供文档迭代接口。

    不负责：
        检索相关性计算、向量编码、结果融合或网络搜索。
    """

    def __init__(self, corpus_path: str | Path) -> None:
        """初始化文档存储。

        输入：
            corpus_path：通过 Dataset.save_to_disk() 保存的语料目录。

        输出：
            无。

        预期行为：
            加载语料，检查必要字段，建立文档 ID 到记录的访问映射。

        异常：
            FileNotFoundError：语料路径不存在。
            ValueError：字段缺失或文档 ID 重复。
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """获取文档总数。

        输入：
            无。

        输出：
            文档数量。
        """
        raise NotImplementedError

    def exists(self, doc_id: str) -> bool:
        """检查文档是否存在。

        输入：
            doc_id：原始文档 ID。

        输出：
            存在返回 True，否则返回 False。
        """
        raise NotImplementedError

    def get(self, doc_id: str) -> Document:
        """获取完整文档。

        输入：
            doc_id：原始文档 ID。

        输出：
            Document，包含 ID、完整正文和来源 URL。

        异常：
            KeyError：文档不存在。
        """
        raise NotImplementedError

    def iter_documents(self) -> Iterator[Document]:
        """逐篇访问文档，供离线建立检索索引使用。

        输入：
            无。

        输出：
            Document 迭代器。
        """
        raise NotImplementedError

    def read(
        self,
        doc_id: str,
        start: int = 0,
        max_chars: int = 4000,
    ) -> DocumentPage:
        """分段读取原文。

        输入：
            doc_id：原始文档 ID。
            start：原文字符起点，从 0 开始。
            max_chars：最多返回的字符数。

        输出：
            DocumentPage，包含原文片段、位置、文档长度及下一读取位置。
            start 等于文档长度时，返回空片段，next_start 为 None。

        异常：
            KeyError：文档不存在。
            ValueError：start 小于 0 或超出文档长度，
                        或 max_chars 不大于 0。
        """
        raise NotImplementedError

    def find(
        self,
        doc_id: str,
        text: str,
        case_sensitive: bool = False,
        max_matches: int = 10,
        context_chars: int = 200,
    ) -> FindResult:
        """在指定文档内查找字面文本。

        输入：
            doc_id：原始文档 ID。
            text：待查找文本，不支持正则表达式。
            case_sensitive：是否区分大小写。
            max_matches：最多返回的匹配数量。
            context_chars：每个匹配前后最多保留的上下文字符数。

        输出：
            FindResult，匹配按原文位置升序排列，采用非重叠匹配。
            所有字符位置对应原始正文。
            未找到时 matches 为空。

        异常：
            KeyError：文档不存在。
            ValueError：text 为空或仅含空白，
                        max_matches 不大于 0，
                        或 context_chars 小于 0。
        """
        raise NotImplementedError


def get_document_store() -> DocumentStore:
    """获取共享的文档存储实例。

    输入：
        无显式参数；语料路径由项目配置提供。

    输出：
        已初始化的 DocumentStore。

    预期行为：
        首次调用时初始化，后续调用复用实例，
        避免每次调用工具都重新加载语料。
    """
    raise NotImplementedError
