# 四人开发任务分工

根据 [项目要求](COMP5423_26Fall_Project.pdf) 和当前仓库骨架，将代码分成四块，每块由一人负责实现和测试。

## 开发者 A：文档与检索模块 (赖悦琳)

**负责代码：**

- `src/search_agent/core/document/`
- `src/search_agent/core/retrieval/`
- `src/search_agent/core/tools/retrieve.py`
- `src/search_agent/core/tools/open_document.py`
- `src/search_agent/core/tools/find_in_document.py`
- `src/search_agent/core/tools/inspect_document.py`

**需要实现：**

1. 加载固定文档集合，根据原始文档 ID 获取全文和来源 URL。
2. 实现分段阅读、词句查找、相关片段选择，以及定位失败后的文档内回退，保留原文偏移。
3. 实现 BM25、encoder 稠密、指令稠密检索的建库/加载/查询，以及 RRF 融合。
4. 接通 `retrieve`、`open_document`、`find_in_document`、`inspect_document`，替换占位返回；Agent 只使用一个统一检索工具。

**交付结果：** Agent 能搜索真实语料、打开检索到的文档，并定位原文中的证据；索引保存后可直接复用。

## 开发者 B：Agent 与上下文管理模块（姚文宇）

**负责代码：**

- `src/search_agent/core/agent.py`
- `src/search_agent/core/config.py`
- `src/search_agent/core/state.py`
- `src/search_agent/core/middlewares/context.py`
- `src/search_agent/core/context/` 和 `core/middlewares/planning.py`
- `.env.example` 和 Agent 工厂相关的导出文件。

**需要实现：**

1. 配置作业指定的 DeepSeek-V4-Flash，接入 A 的工具、C 的技能和 D 的统计 middleware。
2. 使用 LangChain 原生循环，让 Agent 根据已有结果继续检索、调整查询、阅读文档和验证答案。
3. 在聊天历史之外维护已访问文档、选中证据、当前子问题和候选答案。
4. 实现证据去重、预算选择和增量摘要；完善工具消息配对与 TODO 规划衔接，保留完整原始消息供轨迹记录。
5. 提供 CLI 和评估脚本共用的 agent 工厂，支持同步、异步和流式调用；独立评估题之间不共享搜索状态。

**交付结果：** 一个接好各模块、能够多轮搜索并管理上下文的 Agent。

## 开发者 C：技能模块（周彦霖）

**负责代码：**

- `src/search_agent/core/skills/`
- `src/search_agent/core/tools/load_skill.py`
- `src/search_agent/core/tools/skill_resources.py`
- `src/search_agent/core/middlewares/skills.py`

**需要实现：**

1. 扫描技能目录，读取 `SKILL.md` 中的名称和描述。
2. 实现运行时正文加载、资源清单与读取、脚本执行（参数、输出、超时和产物），接通技能工具。
3. 完善“多线索分解”和“搜索恢复”两个技能。
4. 让主模型根据问题或搜索状态自主选择技能，也可以不使用技能。
5. 将当前技能指令注入模型请求，支持切换和卸载；完整正文不永久放入基础提示词。

**交付结果：** 两个可以实际加载和使用的技能，并能展示不同问题触发不同技能选择的过程。

## 开发者 D：轨迹、评估与 CLI 模块（奉翔宇）

**负责代码：**

- `src/search_agent/core/middlewares/trajectory.py`
- `src/search_agent/cli/`
- 新增 `scripts/run_predictions.py`
- 新增 `scripts/evaluate_predictions.py`
- 语料下载、索引构建相关的 `scripts/` 脚本。

**需要实现：**

1. 记录完整、有顺序的模型输出、工具调用、工具结果及技能使用轨迹。
2. 统计 Agent 发起的工具调用次数，以及每题所有 LLM API 响应的实际 token 用量。
3. 完善已有流式 CLI，展示回答、支持证据、选中技能和工具过程，保留连续追问及清空历史功能。
4. 实现验证集、测试集批量运行，每题独立执行，生成预测 JSONL。
5. 实现验证评分，输出 normalized EM、工具调用总数和 token 总数。

**交付结果：** 可交互的完整 CLI，以及包含 `id`、`question`、`answer`、`trajectory`、`tool_calls`、`total_tokens` 六个字段的预测文件；最终测试集每题一条，共 100 条。

## 四人共同完成

- **联调与测试：** 每人完成自己模块的测试，四人共同验证从输入问题到搜索、技能使用、回答和预测输出的完整流程。
- **实验与结果分析：** 共同运行验证集，对比检索、技能和上下文配置，整理真实实验结果。
- **报告：** 四人共同撰写、修改和检查，使用课程模板，正文至少 4 页，完成贡献声明和签名。
- **演示与视频：** 四人共同准备案例、演示系统、录制和剪辑，最终 MP4 不超过 5 分钟。
- **文档与提交：** 四人共同完善 README、复现步骤，检查代码、预测文件、报告、视频和最终提交包。

四块代码可并行开发，先统一工具返回格式和状态字段；模块完成后由四人一起联调。公共的 `state.py`、`config.py` 和 `agent.py` 变更统一交给 B 合入，其他人提供需要的字段和接口。
