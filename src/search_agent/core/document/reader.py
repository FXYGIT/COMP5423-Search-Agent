"""固定文档内的阅读优化与回退骨架，不做外网搜索。"""

from .schemas import InspectionResult
from .store import DocumentStore


class DocumentReader:
    """在 DocumentStore 原文读取之上，定位与问题相关的证据片段。

    优化：查询相关段落选择、相邻命中合并、按字符预算限制返回长度。
    回退：精确查找 → 归一化查找 → 文档内段落排序 → 分页阅读。
    TODO：归一化必须保留到原文偏移的映射；找不到证据时不生成假片段。
    """

    def __init__(self, store: DocumentStore) -> None:
        """输入共用的语料访问层，避免重新加载整库。"""
        self.store = store

    def inspect(self, doc_id: str, query: str, max_chars: int = 4000) -> InspectionResult:
        """输入原始文档 ID、定位问题和长度预算，返回片段、定位方式及续读位置。

        TODO：在单篇原文内依次尝试定位策略；使用 EvidencePassage 返回真实偏移。
        兜底只打开同一固定语料文档，不访问 source_url 获取额外内容。
        """
        raise NotImplementedError
