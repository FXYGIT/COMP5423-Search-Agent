# 第一阶段执行计划：检索模块

## 1. 本阶段要完成什么

让 Agent 能在 BrowseComp-Plus 固定文档集合中搜索、阅读和定位信息。

本阶段只实现数据准备、检索和工具接口。Agent 如何思考、何时继续搜索、如何生成答案，由后续阶段完成。

## 2. 数据从哪里来

使用官方文档集合，共 100,195 篇：

https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus

每篇文档包含文档 ID、完整正文和来源 URL。下载后保存到 `./data/browsecomp_plus_corpus/`。

检索直接使用这些正文，不需要 Web Search，也不需要重新抓取网页。

## 3. 整体工作流程

### 离线准备

```text
下载文档集合
    ↓
读取文档并分块，保留原始文档 ID
    ↓
分别建立三类检索索引
    ├── BM25：稀疏索引
    ├── BGE：普通稠密向量索引
    └── Qwen3-Embedding：指令感知稠密向量索引
    ↓
保存到本地，后续直接加载
```

向量数据库使用 Chroma，保存在运行命令时当前目录的 `./vector_db/` 下。BGE 和 Qwen3-Embedding 使用不同集合。具体模型大小根据设备确定。

### Agent 使用

```text
retrieve：搜索文档，获得文档 ID、分数和片段
    ↓
Agent 判断信息是否足够
    ├── 足够：生成答案
    ├── 想读更多原文：open_document
    ├── 想定位具体词句：find_in_document
    └── 搜索结果不合适：改写查询，再次 retrieve
```

## 4. 目录与职责

```text
src/
├── document/
│   ├── schemas.py                 # 文档和阅读结果的数据结构
│   └── store.py                   # 获取原文、分段阅读、文档内查找
├── retrieval/
│   ├── schemas.py                 # 检索结果的数据结构
│   ├── chunking.py                # 文档分块
│   ├── bm25_retriever.py           # BM25 检索
│   ├── dense_retriever.py          # BGE 检索
│   ├── instruction_retriever.py    # Qwen3-Embedding 检索
│   └── retriever.py                # 统一检索入口，选择方法或融合结果
└── tools/
    ├── __init__.py                 # 导出工具列表
    ├── retrieve.py                 # 统一检索 Tool
    ├── open_document.py            # 原文阅读 Tool
    └── find_in_document.py         # 文档内查找 Tool

scripts/
├── download_corpus.py              # 下载数据集
├── build_sparse_index.py           # 建立 BM25 索引
└── build_vector_db.py              # 分块、编码并写入 Chroma
```

`retrieval/` 写实际检索逻辑；`tools/` 直接使用 LangChain `@tool` 定义工具，调用底层模块，不额外包装工厂函数。

## 5. 需要提供的主要接口

### 文档访问：document/store.py

| 接口 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `get(doc_id)` | 获取完整文档 | 文档 ID | 文档 ID、正文、来源 URL |
| `iter_documents()` | 遍历语料，供建索引使用 | 无 | 文档迭代器 |
| `read(doc_id, start, max_chars)` | 分段阅读原文 | 文档 ID、起点、长度 | 原文片段及下一读取位置 |
| `find(doc_id, text, ...)` | 查找字面文本 | 文档 ID、词句等 | 匹配位置和附近原文 |
| `get_document_store()` | 获取共享实例 | 无 | DocumentStore |

查找位置使用原文字符偏移；找不到文档时报错，找不到词句时返回空匹配。

### 统一检索：retrieval/retriever.py

`retrieve(query, top_k=5, method="hybrid")`

- 输入：查询、返回文档数量、检索方法。
- 方法：`sparse`、`dense`、`instruction_dense`、`hybrid`。
- 输出：按相关性排序的结果，每项包含 `doc_id`、`score`、`snippet`、`source_url`。
- 要求：按原始文档 ID 去重，不能将分块 ID 当作文档 ID。
- 融合：hybrid 内部合并各后端排名，建议使用 RRF，避免直接相加不同模型的原始分数。

### Agent 工具：tools/

| 工具 | 作用 | 调用底层 |
|---|---|---|
| `retrieve` | 搜索整个固定文档集合 | 统一检索入口 |
| `open_document` | 读取指定文档的更多原文 | DocumentStore.read |
| `find_in_document` | 在指定文档中定位词句 | DocumentStore.find |

**Agent 只能看到一个检索工具。三种检索方式在这个工具内部选择或融合，不能分别注册三个检索 Tool。**

## 6. 独立建库脚本要做什么

`build_vector_db.py` 负责：

1. 接收语料路径、模型、集合名称和保存路径。
2. 逐篇读取文档，分块并保留文档 ID、分块 ID、原文位置。
3. 分批计算 embedding，写入 Chroma，避免一次性加载全部分块。
4. 保存模型和分块配置，供后续检索使用。

注意：同一个集合只能使用同一套模型和配置；重复运行不能产生重复记录。检索时加载已有数据库，不重新建库。

## 7. 开发顺序

1. **文档访问**：下载语料，实现 DocumentStore。
2. **最小可用链路**：实现 BM25、统一 retrieve 和 open_document。
3. **向量检索**：完成 Chroma 建库脚本，接入 BGE，再接入 Qwen3-Embedding。
4. **工具补齐**：加入融合和 find_in_document，统一工具输入输出。
5. **联调交付**：提供工具列表、运行配置和 README，交给负责 Agent 的同事。

先用少量文档验证流程，再建立全量索引。

## 8. 完成标准

- 三类检索均能运行，通过同一个 retrieve Tool 调用。
- 检索返回的文档 ID 能打开，片段能在原文中找到。
- open_document 和 find_in_document 可独立调用。
- 索引落盘，程序重启后可直接加载。
- README 说明数据下载、索引构建和工具使用方法。

作业至少要求一类检索；本计划以覆盖三类、争取检索方法满分为目标。工具设计满分还要求至少一种实际可调用的辅助工具。
