"""技能加载接口；builtin 中保留两个项目技能的说明文件。"""

from .loader import SkillLoader
from .schemas import SkillInfo

__all__ = ["SkillLoader", "SkillInfo"]
