from pathlib import Path

from pydantic import BaseModel


class SkillInfo(BaseModel):
    """技能元数据：name/description 来自 YAML frontmatter，directory 指向技能目录。

    模型先看到名称和适用场景，再决定加载哪个技能；此对象不包含完整正文。
    """

    name: str
    description: str
    directory: Path
