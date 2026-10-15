from pathlib import Path

from .schemas import LoadedSkill, SkillInfo, SkillMetadata


class SkillLoader:
    """管理磁盘中的技能目录。"""

    def __init__(self, root: Path) -> None:
        """保存技能根目录，不在构造时加载全部技能正文。"""
        raise NotImplementedError

    def discover(self) -> list[SkillInfo]:
        """扫描 root 下的技能目录。

        要求：
        1. 每个技能目录必须包含 SKILL.md。
        2. 读取并校验 YAML frontmatter。
        3. 返回技能元数据及目录位置。
        4. 不把全部技能正文放入 Agent prompt。
        5. 名称重复或格式错误时给出明确异常。
        """
        raise NotImplementedError

    def load(self, skill: SkillInfo) -> LoadedSkill:
        """读取一个选中技能的 Markdown 正文。

        要求：
        1. 文件必须位于配置的技能根目录内。
        2. 验证正文非空。
        3. 返回正文，不包含 YAML frontmatter。
        """
        raise NotImplementedError

    def _parse_metadata(self, skill_file: Path) -> SkillMetadata:
        """安全解析 YAML，并校验名称、描述和目录名。"""
        raise NotImplementedError
