"""运行时技能发现、正文加载和附属资源的渐进式访问接口。"""

import base64
import re
from pathlib import Path
from typing import Final

from search_agent.core.config import get_config

from .schemas import SkillInfo, SkillResource, SkillResourceContent

_RESOURCE_DIRS: Final[dict[str, str]] = {
    "scripts": "script",
    "references": "reference",
    "assets": "asset",
}
_FRONTMATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)
_FIELD = re.compile(r"^(name|description):\s*(.*?)\s*$")


class SkillLoader:
    """发现技能说明，并按需读取 SKILL.md 正文。

    不另设 manager、selector、session；选择由主模型的工具调用完成，
    当前技能由 SearchState 保存，指令注入交给 SkillMiddleware。
    """

    def __init__(self, root: Path | None = None) -> None:
        """保存技能根目录；此处不读取全部技能正文。"""
        self.root = root if root is not None else get_config().skills.root

    def discover(self) -> list[SkillInfo]:
        """扫描技能子目录，提取 SKILL.md 的 name、description 和目录位置。

        默认技能文件符合规范，不实现重复名称、路径或格式校验。
        """
        if not self.root.is_dir():
            return []
        skills: list[SkillInfo] = []
        for directory in sorted(self.root.iterdir()):
            skill_file = directory / "SKILL.md"
            if not directory.is_dir() or not skill_file.is_file():
                continue
            metadata = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
            name = metadata.get("name")
            description = metadata.get("description")
            if not name or not description:
                raise ValueError(f"{skill_file} must define name and description in frontmatter")
            skills.append(
                SkillInfo(name=name, description=description, directory=directory.resolve())
            )
        return skills

    def list_resources(self, name: str) -> list[SkillResource]:
        """输入技能名称，返回 scripts/references/assets 内文件的相对路径和类型。"""
        directory = self._skill_directory(name)
        resources: list[SkillResource] = []
        for folder, kind in _RESOURCE_DIRS.items():
            resource_root = directory / folder
            if not resource_root.is_dir():
                continue
            for path in sorted(resource_root.rglob("*")):
                if path.is_file() and _is_within(path, directory):
                    resources.append(
                        SkillResource(path=path.relative_to(directory).as_posix(), kind=kind)
                    )
        return resources

    def read_resource(self, name: str, path: str) -> SkillResourceContent:
        """读取技能内相对路径，返回内容与编码。

        路径限制在当前技能目录内，包括符号链接的最终目标。
        """
        directory = self._skill_directory(name)
        resource = _safe_path(directory, path)
        if resource == directory or resource.parent.name not in _RESOURCE_DIRS:
            raise ValueError("resource must be inside scripts, references, or assets")
        if not resource.is_file():
            raise FileNotFoundError(f"skill resource not found: {path}")
        data = resource.read_bytes()
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            content = base64.b64encode(data).decode("ascii")
            encoding = "base64"
        else:
            encoding = "utf-8"
        return SkillResourceContent(
            path=resource.relative_to(directory).as_posix(), content=content, encoding=encoding
        )

    def load(self, name: str) -> str:
        """读取 root/name/SKILL.md 的 Markdown 正文。

        完整技能指令不能永久写入 agent 的基础 system prompt。
        """
        skill_file = _safe_path(self._skill_directory(name), "SKILL.md")
        text = skill_file.read_text(encoding="utf-8")
        match = _FRONTMATTER.match(text)
        if match is None:
            raise ValueError(f"{skill_file} is missing YAML frontmatter")
        return text[match.end() :].lstrip()

    def _skill_directory(self, name: str) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("skill name must be a single directory name")
        for skill in self.discover():
            if skill.name == name:
                return skill.directory
        raise FileNotFoundError(f"skill not found: {name}")


def _parse_frontmatter(text: str) -> dict[str, str]:
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError("SKILL.md is missing YAML frontmatter")
    values: dict[str, str] = {}
    for line in match.group(1).splitlines():
        field = _FIELD.match(line)
        if field:
            value = field.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[field.group(1)] = value
    return values


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not _is_within(candidate, root.resolve()):
        raise ValueError("path escapes the skill directory")
    return candidate


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
