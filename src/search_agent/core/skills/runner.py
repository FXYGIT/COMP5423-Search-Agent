"""技能脚本执行边界；当前只定义协议，不启动进程。"""

from pathlib import Path

from search_agent.core.config import get_config

from .schemas import SkillScriptResult


class SkillScriptRunner:
    """运行技能 scripts/ 下的脚本，并返回 stdout、stderr、退出码与产物。

    TODO：由扩展名选择解释器，参数作为 argv 列表传入，不拼接 shell 命令。
    限定脚本路径在技能目录内，cwd 指向该技能目录，定义输出目录和最小环境；
    超时终止进程并记录 timed_out。该执行器不等于沙箱，隔离策略需另行实现。
    """

    def __init__(self, root: Path | None = None) -> None:
        """记录技能根目录，构造时不加载技能也不执行任何代码。"""
        self.config = get_config().skills
        self.root = root if root is not None else self.config.root

    def run(
        self, name: str, script: str, args: list[str], timeout_seconds: int | None = None
    ) -> SkillScriptResult:
        """输入技能名、相对脚本路径、参数和时限，输出真实执行结果。

        TODO：支持项目运行环境的 Python 脚本；其他解释器按配置显式支持。
        主模型通过 run_skill_script 调用；脚本产生的内部操作不增加 Agent 工具次数。
        timeout_seconds 默认 self.config.timeout_seconds。
        """
        raise NotImplementedError
