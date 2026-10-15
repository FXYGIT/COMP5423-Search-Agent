from collections.abc import Sequence
from typing import Protocol

from .schemas import SkillInfo, SkillSearchContext, SkillSelection


class SkillSelectionModel(Protocol):
    """接入项目统一的 LLM 调用层。

    该调用层负责 API 调用、结构化输出解析和 token 统计。
    """

    async def choose_skill(
        self,
        *,
        question_and_state: SkillSearchContext,
        candidates: Sequence[SkillInfo],
        current_skill_name: str | None,
    ) -> SkillSelection: ...


class SkillSelector:
    """根据当前问题与搜索状态自动选择技能。"""

    def __init__(self, model: SkillSelectionModel) -> None:
        raise NotImplementedError

    async def select(
        self,
        *,
        context: SkillSearchContext,
        available_skills: Sequence[SkillInfo],
        current_skill_name: str | None,
    ) -> SkillSelection:
        """返回一个可用技能名称，或 None。

        要求：
        1. 只给模型提供技能 name、description 和搜索状态摘要。
        2. 允许选择当前技能，表示继续使用。
        3. 允许返回 None，表示不使用技能。
        4. 检查返回名称确实存在，禁止接受虚构名称。
        5. 不按问题 ID、答案或固定题目列表选择。
        """
        raise NotImplementedError
