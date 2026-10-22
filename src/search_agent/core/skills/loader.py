"""运行时技能发现、正文加载和附属资源的渐进式访问接口。"""

from pathlib import Path

from search_agent.core.config import get_config

from .schemas import SkillInfo, SkillResource, SkillResourceContent


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

        TODO：读取 YAML frontmatter，返回元数据列表供主模型选择。
        默认技能文件符合规范，不实现重复名称、路径或格式校验。
        """
        raise NotImplementedError

    def list_resources(self, name: str) -> list[SkillResource]:
        """输入技能名称，返回 scripts/references/assets 内文件的相对路径和类型。

        TODO：按需列出资源；load_skill 应同时返回清单，让主模型知道后续可读取什么。
        """
        raise NotImplementedError

    def read_resource(self, name: str, path: str) -> SkillResourceContent:
        """读取技能内相对路径，返回内容与编码。

        TODO：解析相对路径，文本读取 UTF-8，二进制资产编码 base64；
        路径必须限制在当前技能目录内，包括符号链接的最终目标。
        """
        raise NotImplementedError

    def load(self, name: str) -> str:
        """读取 root/name/SKILL.md 的 Markdown 正文。

        TODO：移除 YAML frontmatter 后返回正文，由 middleware 按轮次注入。
        完整技能指令不能永久写入 agent 的基础 system prompt。
        """
        raise NotImplementedError
