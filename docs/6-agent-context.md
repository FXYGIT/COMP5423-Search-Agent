# Agent 与上下文管理（开发者 B：姚文宇）

B 负责 Agent 工厂、公共配置/状态和搜索上下文。A 的检索/文档工具、C 的技能工具和
D 的轨迹 middleware 由工厂组装；本模块不实现索引、技能加载或评估脚本。
当前 A、C 的业务工具仍可能返回 `not_implemented`，离线模块验证不代表真实搜索已接通。

## 一次请求如何执行

```text
CLI/评估 → create_search_agent → LangChain 原生模型/工具循环
    before_model：处理新增工具结果 → 更新搜索状态 → 达阈值时增量摘要
    请求包装：技能 → TODO 规划 → 上下文选择与配对 → 主模型轨迹 → 模型
    工具结果：完整写入 messages → 下一次 before_model 消费
    after_agent：记录最终候选，并补取 return_direct 工具结果
```

`core/agent.py` 是唯一工厂，返回原生图，支持 `invoke/ainvoke/stream/astream`。
默认模型读取 `AppConfig.llm`，填写服务商提供的 DeepSeek-V4-Flash 模型 ID；不同网关
ID 不一定相同，因此不虚构默认 ID。超时和重试同样来自统一配置。
查询改写、多次检索、阅读及验证由主模型与原生循环完成，没有增加自定义循环或独立规划 LLM。

测试可注入 `model`、`tools`、`system_prompt` 和 `summary_model`。
摘要默认复用主模型，因此生产环境同样使用配置的 DeepSeek-V4-Flash，并产生额外 API 用量。

## 状态与游标

`SearchState.messages` 始终保留完整原始消息。连续追问保留整个返回 state，追加
`HumanMessage` 后再调用；独立评估题使用全新 state。实例仅持有配置和模型，不保存会话记忆，
也不使用全局会话缓存、数据库或 checkpointer。

| 字段 | 含义 |
| --- | --- |
| `visited_doc_ids` | 成功调用文档工具访问的文档 ID；检索命中不等于阅读 |
| `selected_evidence` | 带 `doc_id/text/start/end/source_url` 的原文证据 |
| `active_subqueries` | 未完成 TODO；没有 TODO 时使用最新检索/文档定位查询 |
| `candidate_answer` | 待验证候选，新用户问题清空；最终回答也记录在这里，但不会变成证据 |
| `summary` | 累计研究摘要，仅加入模型请求副本 |
| `search_message_count` | 状态提取已处理的原始消息数 |
| `summarized_message_count` | 累计摘要覆盖的原始消息前缀长度 |
| `trajectory_message_count` | 已写入轨迹的原始消息数，辅助摘要消息不计入此游标 |

三个游标用途不同，不能相互替代。摘要前先补录新到达的原始消息，再写辅助调用轨迹；
此时轨迹游标只按实际补录的原始消息推进。

## 从工具结果提取证据

`context/selection.py::extract_search_progress` 只消费游标后的新消息。
结果必须紧接真实 AI 工具调用，调用 ID、工具名称、文档 ID 都需要匹配。
错误、孤立结果、非法 JSON、`not_implemented` 不会产生证据。
结果优先读取 `ToolMessage.artifact`，否则解析正文 JSON，遵循现有成功信封：

```json
{"status":"ok","result":{"doc_id":"d1","text":"Alice won.","start":0,"end":10,"source_url":"https://example.org/d1"}}
```

`open_document` 使用 `DocumentPage`；`inspect_document` 使用 `passages`；
`find_in_document` 使用 `matches` 中的 `context/context_start/context_end`。
偏移是 Python 原文字符索引，采用 `[start,end)`，要求 `end-start == len(text)`。
检索 `snippet` 没有可验证偏移，技能输出也不是原文，均不会直接成为证据。

`deduplicate_evidence` 合并同文档、来源相容且重叠文本一致的区间。
例如 `[0,6)` 的 `abcdef` 和 `[3,9)` 的 `defghi` 合为 `[0,9)` 的 `abcdefghi`。
不同文档、不同位置、来源冲突或重叠文本冲突会保留，不通过生成式改写来消除冲突。

## TODO 与候选答案

`middlewares/planning.py` 继承原生 `TodoListMiddleware`，保持 `write_todos` 完整列表更新语义。
`pending/in_progress` 条目映射到 `active_subqueries`，已完成条目排除。

新增 `record_candidate_answer(candidate_answer)`，通过 LangGraph `Command` 显式记录搜索中的假设。
调用 ID 由框架注入；传 `null` 清空候选。下一次推理会看到 `UNVERIFIED candidate`，
记录候选不增加证据，也不会自动完成验证任务。

## 预算与模型请求副本

`context/budget.py` 按序列化内容的 UTF-8 字节数除以 3 估算 token，并计入消息和调用开销。
这是无需下载 tokenizer 的启发式估算，不能保证等于 DeepSeek 的精确分词，不能用于评估计费。
实际统计始终来自 API 响应 `usage`。

`SearchContextMiddleware.prepare_request` 在技能和 TODO 注入后，先扣除 system、
OpenAI 兼容格式的工具定义、固定开销和输出预留。输出预留同时作为调用的 `max_tokens`。
`select_context` 在剩余预算中依次保证最新问题、最新完整 AI/tool 块和部分结构化记忆，
再选择近期历史；旧对话按完整用户轮次取舍，当前搜索按完整工具块取舍。
已摘要历史不重复加入，最新问题即使处于已摘要前缀中也完整保留。

记忆优先提供子问题、待验证候选、近期原文证据和摘要，明确标记为不可信数据，不能当成新指令。
超长工具结果仅截短请求副本的正文，并添加截断标记；调用参数、ID、工具结果配对和原始历史保留。
若最新问题或不可裁剪的调用参数仍无法容纳，则抛出明确的预算错误，应增加预算或减少单次参数。
上下文选择结束后再次执行 `repair_tool_messages`，不会伪造结果、补跑中断工具或修改原始消息。

## 增量摘要与真实轨迹

未摘要历史加旧摘要达到阈值时，优先压缩完整旧轮次。单个长问题也可以压缩最新工具块之前的
前缀，但保留最新问题和最新工具块；仅有一个工具块时不单独为问题本身调用摘要。
`ConversationSummarizer` 按完整消息块分批合并旧摘要，每批输入加输出预留均受预算控制。
极长旧消息只在摘要输入副本中裁剪并标明省略，不得据此编造事实。
提示词要求保留文档引用、已知事实、矛盾证据、未决问题、失败查询和 TODO。

旧对话被封装为摘要调用的 user 数据，不能重新作为实际 assistant 响应加入轨迹，
否则旧工具调用和 usage 会重复统计。`SummaryResult.trajectory` 保存本次实际
system/user/assistant 消息，调用处追加后复用 D 的 `trajectory_totals` 重算。
任一响应缺少 usage，总量仍为 `None`，现有评估导出会拒绝不完整统计。

摘要为空或错误地包含工具调用时，保留旧摘要和覆盖游标，记录实际响应用量并写 warning。
传输异常向上传播，不伪造成功。同步使用 `invoke`，异步使用 `ainvoke`；
`nostream` 标签和该次调用的 `stream=False` 防止摘要混入 CLI 回答。
关闭摘要后，状态提取、证据去重和预算裁剪仍然生效。

## 配置

配置统一在 `core/config.py`，环境变量覆盖 `.env`，示例见根目录 `.env.example`。

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `LLM__TIMEOUT_SECONDS` | 120 | 请求超时秒数 |
| `LLM__MAX_RETRIES` | 2 | SDK 重试次数 |
| `CONTEXT__MAX_TOKENS` | 16000 | 包括 system、工具定义及输出预留的总估算预算 |
| `CONTEXT__SUMMARY_TRIGGER_TOKENS` | 12000 | 历史摘要触发阈值，建议低于可用输入预算 |
| `CONTEXT__RESERVE_OUTPUT_TOKENS` | 2000 | 主模型输出预留及调用输出上限 |
| `CONTEXT__EVIDENCE_MAX_TOKENS` | 4000 | 记忆中证据文本的额外上限 |
| `CONTEXT__SUMMARY_MAX_TOKENS` | 1500 | 摘要输出预留及累计摘要大小上限 |
| `CONTEXT__SUMMARY_ENABLED` | true | 是否调用摘要模型 |

## 验证与团队联调

```bash
uv sync --locked
uv run pre-commit install
uv run pytest tests/test_context.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

`tests/test_context.py` 使用假模型和离线工具运行真实 LangChain 图，覆盖四种调用模式、
原文提取、无效结果、区间去重、游标幂等、完整轮次/工具块、超长结果、预算错误、
TODO/候选衔接、分批摘要、CLI 摘要隔离、真实 usage 汇总和跨题状态隔离。

等待 A、C 工具实现并配置真实 API 后，可进行人工联调：

1. `uv run search-agent --plain --verbose` 提问需要多次文档核对的问题；预期显示真实工具过程，
   `/evidence` 显示文档证据，重复阅读不重复添加相同片段。
2. `/todos` 查看任务并连续追问；预期保留搜索进度，已完成任务不再出现在待处理子问题中。
3. 临时调低摘要阈值后执行长搜索；预期能继续回答，终端不泄漏内部摘要，state/轨迹记录实际辅助用量。
4. `/clear` 后问无关问题；预期文档、证据、摘要、候选及 TODO 从空状态开始。
   验证后恢复配置。预算或 API 异常查看 `logs/search_agent.log`，同时保留工具参数和失败信息。

本次不调用真实 API；真实 DeepSeek、语料/技能和完整批量评估仍需团队集成验证。
