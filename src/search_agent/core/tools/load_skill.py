"""主模型选择技能的工具入口，不增加专用的技能选择模型。"""

from typing import Any

from langchain.tools import tool


@tool
def load_skill(name: str) -> dict[str, Any]:
    """按名称加载技能，供当前搜索步骤使用。

    规划技能：multi-clue-decomposition 用于多线索分解，
    search-recovery 用于搜索失败后的查询调整。
    TODO：调用 SkillLoader.load/list_resources，返回名称、正文、资源清单；
    通过 Command 更新 active_skill/skill_instructions 和配对 ToolMessage，
    由 SkillMiddleware 注入下一次请求。成功结果为 {status: "ok", result: {...}}。
    """
    return {"status": "not_implemented", "message": "技能加载尚未实现。", "result": None}
