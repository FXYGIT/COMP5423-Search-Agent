from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class SkillInfo(BaseModel):
    """技能元数据：name/description 来自 YAML frontmatter，directory 指向技能目录。

    模型先看到名称和适用场景，再决定加载哪个技能；此对象不包含完整正文。
    """

    name: str
    description: str
    directory: Path


class SkillResource(BaseModel):
    """技能内相对路径及类型；发现阶段只返回清单，不读取全部资源。"""

    path: str
    kind: Literal["script", "reference", "asset"]


class SkillResourceContent(BaseModel):
    """资源读取结果；文本用 UTF-8，二进制资产用 base64，不把二进制直接放进 prompt。"""

    path: str
    content: str
    encoding: Literal["utf-8", "base64"]


class SkillScriptResult(BaseModel):
    """执行的实际结果；未执行时不能伪造 exit_code=0。

    artifacts 是脚本产生的文件路径；展示 stdout/stderr 不表示它们自动成为证据。
    """

    skill: str
    script: str
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    artifacts: list[str]
