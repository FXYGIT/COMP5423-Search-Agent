# Core 扩展骨架与流式 CLI

当前目标是“模块边界完整、调用流程可运行”。核心业务算法只保留职责、输入输出与
TODO；CLI、原生任务规划和工具消息配对有最小实际实现。默认业务输入合法。

| 模块 | 当前行为 | 后续 TODO |
| --- | --- | --- |
| agent.py / config.py | 统一读取 .env 中的各模块配置，组装原生 LangChain agent | 随业务实现接入组件实例 |
| logging_setup.py | 设置只有 FileHandler 的 root logger，统一日志格式 | 无控制台日志输出 |
| state.py | 完整结构化状态协议，CLI 跨轮保留 | 由各模块填充搜索进度、摘要与统计 |
| context/messages.py | 修复模型请求视图的悬空工具调用和孤立结果 | 与未来裁剪/摘要流程协作 |
| context/summary.py | 摘要结果与调用接口 | 增量摘要、记录实际 API 用量 |
| context/selection.py | 状态提取与预算选择接口 | 去重证据、选择完整轮次、预算裁剪 |
| middlewares/context.py | 同步/异步配对；before_model 返回空更新 | 进度提取、摘要、上下文选择 |
| middlewares/skills.py | 同步/异步透传 | 目录元数据与当前技能正文按需注入 |
| middlewares/planning.py | 复用 TodoListMiddleware 与 write_todos | 技能指导子问题拆解与进度维护 |
| middlewares/trajectory.py | 同步/异步空更新 | 全轨迹、工具次数、全部 API token 统计 |
| skills/loader.py | 发现、正文、资源清单与读取接口 | YAML 元数据、渐进式加载 |
| skills/runner.py | 脚本执行输入输出协议 | 解释器、argv、cwd、超时、stdout/stderr/产物 |
| cli/main.py / display.py | 多轮流、状态保存、Rich 展示与交互命令 | 随真实业务数据接入验证 |

四个 middleware 均已注册；未实现的钩子透传，不调用会抛出 NotImplementedError 的业务方法。
配置统一在 core/config.py 定义，覆盖 LLM、embedding、Chroma、检索、建库、文档、技能、
上下文、评估和日志。导入 core 不读取 .env 或加载业务数据；运行入口按需读取配置快照。
只执行离线复评时无需 LLM 密钥。所有运行日志写文件，CLI 负责终端交互。

## 工具协议

只有一个 retrieve 工具提供四种内部模式。文档辅助工具为 open_document、
find_in_document、inspect_document。技能工具为 list_skills、load_skill、
read_skill_resource、run_skill_script、unload_skill，规划工具 write_todos 由原生 middleware 提供。

业务工具当前返回 `{status: "not_implemented", message: ..., result: null}`。
成功接通后使用 `{status: "ok", result: ...}`，具体结果类型见对应函数注释。
技能脚本工具参数为 name、script、arguments、timeout_seconds；
arguments 是 argv 列表。当前不运行进程，不虚报 exit_code=0。

load_skill 的 TODO 包括更新 active_skill/skill_instructions 和返回资源清单；
资源按技能相对路径读取。unload_skill 清空当前技能。
完整技能正文不永久写进基础 prompt，脚本执行结果与真实原文证据需要区分。

## 状态与上下文

messages 保留完整原始过程；模型请求视图单独处理。
当前配对器保留已返回的 AI/tool 对，删除未返回的调用和孤立结果，不制造 ToolMessage，
不重复执行被中断的工具，不修改原始消息。它并不表示已完成摘要或上下文压缩。

后续 summary 保存累计摘要，summarized_message_count 标记覆盖范围；
摘要只压缩完整旧轮次，并保留事实来源、未决问题和任务。
所有摘要调用也必须计入真实 API 用量与轨迹。

## CLI

`uv run search-agent` 启动；`--plain` 输出纯文本，`--verbose` 展示完整工具结果。

- messages：逐段输出模型文本，立即 flush。
- updates：显示完整工具参数及执行结果，不重复打印流式文字。
- values：保存完整 SearchState，显示任务、技能及证据变化。

工具长结果只截短终端预览，原始消息保持完整。
/help、/todos、/skills、/evidence、/status 查看相应内容；/clear 清空完整状态；
/exit、exit、quit 退出。生成中 Ctrl+C 中断本轮，输入时 Ctrl+C/Ctrl+D 退出。
退出后不持久化会话，不引入额外数据库或会话框架。
未实现的统计显示“未统计”，没有证据时显示空状态。

CLI 和网关共用 create_search_agent。网关下一轮应保留完整返回状态，并在 messages
后追加新 HumanMessage；每个会话、每道独立评估题分别维护状态。

## 验证边界

离线测试使用真实 LangChain 图和假模型，验证流式输出、工具执行前展示、
完整状态保留与清空、同步/异步工具循环、占位结果和消息配对。
这不等于真实检索、技能脚本、摘要或评分管线已经实现，也不代替真实模型 API 验证。
