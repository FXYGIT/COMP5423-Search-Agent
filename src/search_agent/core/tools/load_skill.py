"""主模型选择技能的工具入口，不增加专用的技能选择模型。"""

from typing import Annotated, Any

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from search_agent.core.skills import SkillLoader


@tool
def load_skill(name: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """按名称加载技能，供当前搜索步骤使用。

    规划技能：multi-clue-decomposition 用于多线索分解，
    search-recovery 用于搜索失败后的查询调整。
    成功时通过 Command 更新 active_skill/skill_instructions，并把结果写入配对
    ToolMessage；SkillMiddleware 只在后续请求中注入正文。
    """
    try:
        loader = SkillLoader()
        instructions = loader.load(name)
        resources = loader.list_resources(name)
    except (OSError, ValueError, FileNotFoundError) as error:
        result: dict[str, Any] = {"status": "error", "message": str(error), "result": None}
        return Command(
            update={
                "active_skill": None,
                "skill_instructions": "",
                "messages": [ToolMessage(str(result), tool_call_id=tool_call_id)],
            }
        )
    result = {
        "status": "ok",
        "result": {
            "name": name,
            "instructions": instructions,
            "resources": [resource.model_dump(mode="json") for resource in resources],
        },
    }
    return Command(
        update={
            "active_skill": name,
            "skill_instructions": instructions,
            "messages": [ToolMessage(str(result), tool_call_id=tool_call_id)],
        }
    )
