# MVP 检索模块实现说明

本轮按最小 MVP 收敛，先保留一类检索的接口骨架，暂不实现算法。
原先三类检索与向量库融合属于可选扩展，不是当前 core 的运行路径。

## 数据

使用 BrowseComp-Plus 的固定文档集合。下载脚本将文档保存到
`data/browsecomp_plus_corpus/`，每篇文档具有原始 ID、正文和来源 URL。
检索只使用固定语料，不需要网络搜索。

## 核心接口

| 位置 | 职责 | 待实现内容 |
| --- | --- | --- |
| document/store.py | 加载和读取语料 | 按 ID 获取全文、分段阅读、字面查找 |
| retrieval/retriever.py | 统一检索 | 构建/加载 BM25 索引，按查询排序 |
| retrieval/schemas.py | 检索结果 | doc_id、score、snippet、source_url |
| tools/retrieve.py | Agent 唯一检索工具 | 调用 Retriever 并序列化结果 |
| tools/open_document.py | 辅助工具 | 调用 DocumentStore.read |
| tools/find_in_document.py | 辅助工具 | 调用 DocumentStore.find |

## 后续实现顺序

1. 在 DocumentStore 加载本地语料，建立原始 ID 到文档的映射。
2. 在 Retriever.build_index 中分词、统计词频与文档频率并保存 BM25 索引。
3. 在 Retriever.retrieve 中查询、排序，返回 top_k 篇文档及证据片段。
4. 替换三个工具的 not_implemented 返回，连接文档和检索接口。
5. 用小型固定语料验证检索 ID 能打开，再在完整语料上运行。

所有输入默认合法，不实现额外校验和异常恢复。
不增加多个检索工具；以后扩展检索方式仍由同一个 retrieve 入口暴露。

现有 `scripts/build_vector_db.py` 是此前保留的可选向量建库草稿，
不在当前 MVP 中执行，BM25 实现无需它。
