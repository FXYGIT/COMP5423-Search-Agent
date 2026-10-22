"""技能的渐进式资源访问与脚本执行入口；当前均明确返回占位状态。"""

from typing import Any

from langchain.tools import tool


@tool
def list_skills() -> dict[str, Any]:
    """列出可用技能名称与适用场景，供主模型自主选择。

    TODO：SkillLoader.discover → {status: "ok", result: list[SkillInfo]}。
    仅返回元数据，不在此加载所有技能全文。
    """
    return {"status": "not_implemented", "message": "技能发现尚未实现。", "result": None}


@tool
def read_skill_resource(name: str, path: str) -> dict[str, Any]:
    """按技能名和相对路径读取 references、assets 或 scripts 中的文件。

    TODO：SkillLoader.read_resource → {status: "ok", result: SkillResourceContent}。
    阅读脚本不会执行脚本，二进制资源按 encoding 处理。
    """
    return {"status": "not_implemented", "message": "技能资源读取尚未实现。", "result": None}


@tool
def run_skill_script(
    name: str, script: str, arguments: list[str], timeout_seconds: int = 30
) -> dict[str, Any]:
    """执行已加载技能的 scripts/ 内脚本，参数使用列表而非 shell 命令。

    arguments 是脚本 argv 列表。TODO：SkillScriptRunner.run(name, script, arguments,
    timeout_seconds) → {status: "ok", result: SkillScriptResult}；
    输出 stdout、stderr、exit_code、timed_out、artifacts，由模型检查实际是否成功。
    当前不创建进程，不执行任何脚本，也不声称已成功。
    """
    return {"status": "not_implemented", "message": "技能脚本执行尚未实现。", "result": None}


@tool
def unload_skill() -> dict[str, Any]:
    """结束当前技能，避免后续搜索继续携带无关指令。

    TODO：通过 Command 更新 active_skill=None、skill_instructions=""，
    同时写入与本次调用配对的 ToolMessage；返回 {status: "ok", result: None}。
    """
    return {"status": "not_implemented", "message": "技能卸载尚未实现。", "result": None}
