# 检索与文档扩展骨架

固定语料为 BrowseComp-Plus。当前仅定义接口和 TODO，不加载模型、构建索引或生成假证据。

| 文件 | 职责与输入输出 |
| --- | --- |
| retrieval/bm25.py | BM25：语料→索引；query/top_k→排序文档 |
| retrieval/dense.py | Encoder 稠密检索：模型专属向量索引；query→文档 |
| retrieval/instruction.py | LLM-based 指令嵌入：instruction + query→文档 |
| retrieval/fusion.py | RRF：多路排名→按原始 doc_id 去重的融合排名 |
| retrieval/retriever.py | 统一 build/load/retrieve，内部方法路由与融合 |
| document/store.py | 原始 doc_id→全文、分页、字面查找 |
| document/reader.py | 文档 ID + query→相关片段；定位失败时逐级回退 |
| document/schemas.py | 原文、分页、匹配、证据和定位结果协议 |

Agent 只获得一个 `retrieve(query, top_k=5, method="hybrid", instruction=...)` 工具：
method 为 sparse、dense、instruction 或 hybrid。每类后端维护独立索引及模型配置；
hybrid 内部融合三路排名，不直接相加不同方法的原始分数。

检索统一返回 `list[RetrievalResult]`，每项含 doc_id、score、snippet、source_url。
向量块命中要映射回原始文档 ID，再聚合文档排名，使阅读工具能打开对应原文。
索引构建离线完成，查询时复用已加载模型与索引。

文档回退在同一篇固定语料原文内进行：精确定位→归一化定位→段落排序→顺序分页。
`InspectionResult` 返回采用的方法、原文片段与偏移、next_start；
归一化或段落排序不能破坏原文坐标，也不访问来源 URL 补充外网内容。

实现顺序：DocumentStore → 三类检索与各自建库 → RRF/统一路由 → 阅读优化 →
接通 retrieve/open_document/find_in_document/inspect_document。
现有 scripts/build_vector_db.py 仍为可选建库草稿，未接入运行。

所有业务方法抛出 NotImplementedError；工具目前返回 not_implemented。
仅实际搜索完成而没有命中时，才能返回成功状态和空列表。

题目第 1.1 节最低要求一类检索，三类全部实现可得该项满分；融合为可选方案。
文档回退是本项目扩展，并非题目指定的独立硬性功能。
