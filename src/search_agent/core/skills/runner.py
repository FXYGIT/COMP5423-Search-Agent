"""技能脚本执行边界。"""

import os
import subprocess
import sys
from pathlib import Path

from search_agent.core.config import get_config

from .schemas import SkillScriptResult


class SkillScriptRunner:
    """运行技能 scripts/ 下的 Python 脚本，不经过 shell。"""

    def __init__(self, root: Path | None = None) -> None:
        """记录技能根目录，构造时不加载技能也不执行任何代码。"""
        self.config = get_config().skills
        self.root = root if root is not None else self.config.root

    def run(
        self, name: str, script: str, args: list[str], timeout_seconds: int | None = None
    ) -> SkillScriptResult:
        """输入技能名、相对脚本路径、参数和时限，输出真实执行结果。

        主模型通过 run_skill_script 调用；脚本产生的内部操作不增加 Agent 工具次数。
        timeout_seconds 默认 self.config.timeout_seconds。
        """
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if any(not isinstance(argument, str) for argument in args):
            raise TypeError("script arguments must be strings")
        skill = self._skill_directory(name)
        script_path = (skill / script).resolve()
        try:
            relative_script = script_path.relative_to(skill)
        except ValueError:
            raise ValueError("script must be inside the skill scripts directory") from None
        if not relative_script.parts or relative_script.parts[0] != "scripts":
            raise ValueError("script must be inside the skill scripts directory")
        if not script_path.is_file():
            raise FileNotFoundError(f"skill script not found: {script}")
        if script_path.suffix.lower() != ".py":
            raise ValueError("only Python skill scripts are supported")

        before = _snapshot(skill)
        command = [sys.executable, str(script_path), *args]
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
        }
        timeout = timeout_seconds or self.config.timeout_seconds
        try:
            completed = subprocess.run(
                command,
                cwd=skill,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return SkillScriptResult(
                skill=name,
                script=script,
                stdout=_output(error.stdout),
                stderr=_output(error.stderr),
                exit_code=None,
                timed_out=True,
                artifacts=_changed_files(skill, before),
            )
        return SkillScriptResult(
            skill=name,
            script=script,
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            timed_out=False,
            artifacts=_changed_files(skill, before),
        )

    def _skill_directory(self, name: str) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("skill name must be a single directory name")
        directory = (self.root / name).resolve()
        if not _is_within(directory, self.root.resolve()) or not (directory / "SKILL.md").is_file():
            raise FileNotFoundError(f"skill not found: {name}")
        return directory


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _snapshot(root: Path) -> dict[str, int]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("*")
        if path.is_file()
    }


def _changed_files(root: Path, before: dict[str, int]) -> list[str]:
    after = _snapshot(root)
    return sorted(path for path, stamp in after.items() if before.get(path) != stamp)


def _output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value
