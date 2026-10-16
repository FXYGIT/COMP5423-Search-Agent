# Core 最小骨架与流式 CLI

## 当前范围

按本轮要求，业务算法和 middleware 的具体实现已移除，只保留详细职责说明、
函数输入输出与 TODO。默认输入合法，不做参数校验、重试、预算限制或会话管理框架。
真正运行的部分只有模型配置、LangChain agent 组装、工具占位入口和 CLI。

这是一套便于继续开发的 MVP 骨架，不代表已完成作业的检索、技能、上下文或评估功能。

## 目录职责

```text
core/
  agent.py                 # 创建模型和 LangChain agent，CLI/网关共用
  config.py                # 从 .env 读取模型 API 配置
  state.py                 # 作业结构化状态草稿，尚未接入
  document/
    schemas.py             # 原文、阅读片段、文档内匹配的数据结构
    store.py               # 原文读取/查找接口 TODO
  retrieval/
    schemas.py             # 统一检索结果
    retriever.py           # BM25 建索引及检索接口 TODO
  skills/
    schemas.py             # 技能 name、description、directory
    loader.py              # 发现技能和按需读取正文 TODO
    builtin/               # 两个技能的 Markdown 说明
  tools/
    retrieve.py            # 唯一检索工具
    open_document.py       # 原文阅读工具
    find_in_document.py    # 文档内查找工具
    load_skill.py          # 模型选择并加载技能的工具
  middlewares/
    context.py             # 搜索状态与上下文管理 TODO
    skills.py              # 当前技能指令注入 TODO
    trajectory.py          # 轨迹及调用/token 统计 TODO
cli/
  main.py                  # 流式显示、多轮历史和交互命令
  __main__.py              # python -m search_agent.cli
```

## 作业要求如何落地

- **检索与工具**：作业至少需要一类检索，MVP 选择 BM25。只保留一个 Retriever，
  暂不设计多后端抽象或融合。工具目前返回明确的 `not_implemented`。
- **迭代循环**：由 LangChain `create_agent` 原生处理模型→工具→模型，不自行实现第二套循环。
- **上下文管理**：保留 SearchState 和 SearchContextMiddleware 骨架。未来记录已读文档、
  选中证据、子查询和候选答案，并对模型请求去重/裁剪。CLI 当前传递完整历史，不声称完成 Feature B。
- **可复用技能**：保留 SkillLoader、load_skill、SkillMiddleware 和两个 SKILL.md。
  TODO 是运行时读取和注入。主模型自行选择技能，不引入 selector、manager、session。
- **评估统计**：TrajectoryMiddleware 保留逐次 API 用量和最终过程整理的接口。
  当前没有生成提交 JSONL，也不展示虚假的 token/轨迹统计。
- **界面**：CLI 已完整实现文本流、工具调用/结果展示和连续追问。

三个 middleware 都不注册到当前 agent，避免运行未实现的钩子。
工具中的占位返回是演示调用链所需的最少代码，不能算作真实搜索或技能执行。

## CLI 如何流式工作

`stream_turn` 同时订阅 LangChain 的 `messages` 和 `updates`：

1. `messages`：拿到模型文本片段后立即打印并 flush。
2. `updates`：模型完成一次工具选择后打印工具名称及完整参数；
   工具执行完成后立即打印返回内容。
3. 将 updates 中的完整模型消息和工具消息加入历史，下一轮带上历史继续推理。
4. 不在 updates 中重复打印已流式输出过的文本。

工具参数在完整生成后显示，不对不完整的 JSON token 做解析。
工具结果按执行完成的更新显示，不等整个 agent 结束才集中输出。

不使用 checkpointer；CLI 在内存中保存 messages，`/clear` 直接清空。
退出程序后不保存聊天记录。网关可调用同一工厂，并自行保存每个会话的 messages。

## 已删除的设计

预算与异常恢复 middleware、日志文件管理、技能 manager/selector/session、
多后端检索抽象，以及路径校验、名称校验、元数据校验等边界逻辑。
DeepAgents、Tavily、直接 dotenv/PyYAML 依赖已经移除。

## 验证

离线测试使用真实 LangChain agent 图和脚本流式模型，验证连续追问、清空历史、
文本在生成过程中显示、工具调用先于执行显示，以及占位结果不冒充真实证据。
不需要调用真实模型 API。

API 参考：[LangChain stream modes](https://reference.langchain.com/python/langgraph/types/StreamMode)。
