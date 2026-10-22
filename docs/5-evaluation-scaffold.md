# 评估模块骨架

流程：BrowseComp-Plus 固定语料 → embedding/分块 → 当前目录的 `./chroma` →
读取本地验证集 → 每题独立启动一次 Agent 运行 → `test_prediction.jsonl` → 指标和报告。
Agent 工厂可复用，但每题传入全新的 state；参考答案只用于运行后的评分。

本轮实现轻量流程接线、数据读写、normalized EM 和报告输出。
embedding 创建、Chroma 工具适配、索引构建、完整轨迹统计仍为明确的 TODO。
这些接口补齐前，默认预测命令会报告尚未实现，不创建伪造预测或填入虚假的 0 token。

模块约定：

- `core/config.py`：唯一配置定义及 .env 入口，评估不再另设配置类。
- `evaluation/schemas.py`：问题、六字段预测记录和报告。
- `evaluation/dataset.py`：读取本地 Hugging Face 数据集，分离参考答案。
- `core/retrieval/chroma.py`：共享 embedding 与 Chroma 打开接口。
- `evaluation/chroma.py`：将统一检索器与原文/技能绑定到评估工具的 TODO 接口。
- `evaluation/runner.py`：组装 Agent、逐题执行、提取结果、JSONL 读写。
- `evaluation/metrics.py`：按 ID 对齐，计算 normalized EM 和调用/token 总量。
- `evaluation/report.py`：输出 JSON 与 Markdown 报告。
- `scripts/build_vector_db.py`：离线建库接口；默认输出 `./chroma`。
- `scripts/run_predictions.py`：完整评估命令入口。
- `scripts/evaluate_predictions.py`：对已有预测文件重新评分，无需模型或 Chroma。

建库、检索和评估通过同一 AppConfig 使用相同 embedding 模型、编码方式和 collection。
`INDEXING__CATEGORY` 选择 encoder/instruction 建库；`RETRIEVAL__METHOD` 控制查询路由，
hybrid 需要两类向量集合和 BM25。模型、集合不能在评估脚本中重新定义。
Chroma 保存 chunk 原文和 `doc_id/chunk_id/start/end/source_url` 元数据；普通稠密与
指令嵌入使用不同 collection。原始语料也保留在本地，供全文工具读取。
评估仅打开已有库，不在运行测试题时自动下载语料、建索引或创建空集合。

预测记录严格包含 `id/question/answer/trajectory/tool_calls/total_tokens`。
trajectory 由 TrajectoryMiddleware 提供完整时间顺序记录，包括摘要等额外 LLM 调用；
token 使用 API 实际 usage。统计未实现时直接指出缺少接口，不从字符数推算。
最终 answer 应由 Agent 输出简短答案，不能把解释、引用或状态提示当作标准答案匹配。

指标包含题数、normalized EM、工具调用总数/均值、API token 总数/均值及逐题结果。
预测 ID 必须唯一且完整对应当前评估数据；不能仅对成功或已预测的题目计算分数。
无参考答案的测试集仍输出用量和覆盖情况，EM 为 null；不推算班级相对排名分数。
验证集运行结果不是最终测试集提交文件，即使文件名同为 test_prediction.jsonl。

```bash
# 先在 .env 配好模型和路径；建库/完整评估仍需补齐业务 TODO
uv run python scripts/build_vector_db.py
uv run python scripts/run_predictions.py

# 默认读取 ./data/browsecomp_plus_val 的 validation split
# 输出 ./test_prediction.jsonl、./evaluation_report.json、./evaluation_report.md
# 已有预测文件可直接评分，不运行模型
uv run python scripts/evaluate_predictions.py

# 小样本联调：在 .env 设置 EVALUATION__LIMIT=3，再运行同一命令
# 正式运行将 LIMIT 留空；报告记录实际题数，不代表完整 100 题提交
```

配置项：`EVALUATION__DATASET_PATH`、`SPLIT`、`PREDICTIONS_PATH`、`OUTPUT_DIR`、`LIMIT`。
命令不再接受配置参数。运行状态和失败原因只写 `LOGGING__PATH`（默认 logs/search_agent.log），
失败返回非零退出码；预测文件与报告仍写入各自配置的输出路径。

实现顺序：完成 DocumentStore/embedding/建库 → Chroma 工具适配 →
TrajectoryMiddleware 的完整轨迹与真实用量 → 运行验证集并检查报告。
这些职责沿用原分工；本轮不下载模型、不运行真实 API 批量评估。
