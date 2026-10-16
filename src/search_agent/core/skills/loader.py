"""运行时技能读取接口，保留作业要求所需的两个操作。"""

from pathlib import Path

from .schemas import SkillInfo


class SkillLoader:
    """发现技能说明，并按需读取 SKILL.md 正文。

    不另设 manager、selector、session；选择由主模型的工具调用完成，
    当前技能由 SearchState 保存，指令注入交给 SkillMiddleware。
    """

    def __init__(self, root: Path) -> None:
        """保存技能根目录；此处不读取全部技能正文。"""
        self.root = root

    def discover(self) -> list[SkillInfo]:
        """扫描技能子目录，提取 SKILL.md 的 name、description 和目录位置。

        TODO：读取 YAML frontmatter，返回元数据列表供主模型选择。
        默认技能文件符合规范，不实现重复名称、路径或格式校验。
        """
        raise NotImplementedError

    def load(self, name: str) -> str:
        """读取 root/name/SKILL.md 的 Markdown 正文。

        TODO：移除 YAML frontmatter 后返回正文，由 middleware 按轮次注入。
        完整技能指令不能永久写入 agent 的基础 system prompt。
        """
        raise NotImplementedError
