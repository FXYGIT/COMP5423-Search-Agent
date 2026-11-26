"""技能的渐进式资源访问与脚本执行入口。"""

from typing import Annotated, Any

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from search_agent.core.skills import SkillLoader, SkillScriptRunner


@tool
def list_skills() -> dict[str, Any]:
    """列出可用技能名称与适用场景，供主模型自主选择。

    仅返回元数据，不在此加载所有技能全文。
    """
    try:
        skills = SkillLoader().discover()
    except (OSError, ValueError) as error:
        return {"status": "error", "message": str(error), "result": None}
    return {
        "status": "ok",
        "result": [
            {
                "name": skill.name,
                "description": skill.description,
            }
            for skill in skills
        ],
    }


@tool
def read_skill_resource(name: str, path: str) -> dict[str, Any]:
    """按技能名和相对路径读取 references、assets 或 scripts 中的文件。

    阅读脚本不会执行脚本，二进制资源按 encoding 处理。
    """
    try:
        resource = SkillLoader().read_resource(name, path)
    except (OSError, ValueError) as error:
        return {"status": "error", "message": str(error), "result": None}
    return {"status": "ok", "result": resource.model_dump(mode="json")}


@tool
def run_skill_script(
    name: str,
    script: str,
    arguments: list[str],
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """执行已加载技能的 scripts/ 内脚本，参数使用列表而非 shell 命令。

    arguments 是脚本 argv 列表；
    输出 stdout、stderr、exit_code、timed_out、artifacts，由模型检查实际是否成功。
    timeout_seconds 默认使用 SKILLS__TIMEOUT_SECONDS。
    """
    try:
        result = SkillScriptRunner().run(name, script, arguments, timeout_seconds)
    except (OSError, TypeError, ValueError) as error:
        return {"status": "error", "message": str(error), "result": None}
    return {"status": "ok", "result": result.model_dump(mode="json")}


@tool
def unload_skill(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """结束当前技能，避免后续搜索继续携带无关指令。

    通过 Command 清空 active_skill/skill_instructions，并写入配对的 ToolMessage。
    """
    result = {"status": "ok", "result": None}
    return Command(
        update={
            "active_skill": None,
            "skill_instructions": "",
            "messages": [ToolMessage(str(result), tool_call_id=tool_call_id)],
        }
    )
