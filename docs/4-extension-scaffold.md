# 扩展骨架说明

本轮交付可运行的开发骨架。核心算法只定义职责、参数、返回结构和 TODO；CLI 与必要的运行接线实际实现。不把占位结果当成已完成的作业功能。

| 模块 | 本轮接口 | 实现边界 |
| --- | --- | --- |
| retrieval | BM25、encoder dense、instruction dense、RRF 融合与统一路由 | 算法、模型加载、索引保存和搜索均 TODO；仅暴露一个 retrieve 工具 |
| document | 原文读取、分页、查找、片段选择与定位回退 | 固定语料内优化；不引入外网补充证据，业务算法 TODO |
| skills | 目录发现、正文加载、资源读取、脚本执行、卸载 | 脚本参数、输出、退出码和超时结构明确；当前不执行脚本，工具明确返回 not_implemented |
| context | 结构化证据、摘要、预算选择、工具消息配对 | 摘要与裁剪算法 TODO；模型请求副本中移除未完成的工具调用，原始历史保留 |
| planning | todos 与 write_todos | 复用 LangChain TodoListMiddleware，能够实际更新任务列表 |
| middleware | 上下文、技能、规划、轨迹 | 主 Agent 轨迹/usage 已接入；其他未实现 hook 透传，同步/异步一致 |
| CLI | 流式文字、工具卡片、任务、技能、证据、状态 | 实际实现；保留完整 state，支持 /help、/clear、/todos、/skills、/evidence、/status 与 --plain、--verbose |

题目第 1.1 节要求至少一类检索，三类全部实现可获得该项满分；融合是可选内部设计。第 1.3 节中 scripts/references/assets 为可选目录。文档回退、悬空工具调用处理和任务规划是本项目补充能力。只有真实实现的功能才能在报告中主张得分。

实现顺序：先补核心接口和状态，接入可透传的 middleware，再实现 CLI 的状态消费与展示，最后更新说明并验证。验证关注真实流式输出、工具执行前展示、跨轮任务保留、占位工具可调用、同步/异步一致、工具调用配对和清空会话。

四人分工继续沿用 [开发任务分工](3-team-development-plan.md)：A 负责文档/检索，B 负责 Agent/上下文/规划，C 负责技能，D 负责轨迹/评估/CLI；报告与演示共同完成。

参考：[Agent Skills 规范](https://agentskills.io/specification)、[LangChain middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)。具体 API 以仓库锁定的 LangChain 1.4.6 为准。
