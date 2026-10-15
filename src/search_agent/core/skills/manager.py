from collections.abc import Callable

from .loader import SkillLoader
from .schemas import (
    SkillEvent,
    SkillInfo,
    SkillSearchContext,
    SkillSelection,
    SkillSession,
)
from .selector import SkillSelector


class SkillManager:
    """协调技能发现、选择、加载和指令提供。

    不负责文档检索，也不负责直接生成最终答案。
    """

    def __init__(
        self,
        *,
        loader: SkillLoader,
        selector: SkillSelector,
        record_event: Callable[[SkillEvent], None],
    ) -> None:
        """注入依赖。

        record_event 接入主 Agent 的统一 trajectory，
        不另建一套互不关联的轨迹文件。
        """
        raise NotImplementedError

    def initialize(self) -> list[SkillInfo]:
        """启动时发现技能，保存可用目录与元数据。"""
        raise NotImplementedError

    def create_session(self, question_id: str) -> SkillSession:
        """为一个问题创建独立技能状态。"""
        raise NotImplementedError

    async def update(
        self,
        *,
        session: SkillSession,
        context: SkillSearchContext,
        step: int,
    ) -> SkillSelection:
        """根据当前状态选择并更新技能。

        处理规则：
        - 选中当前技能：继续使用，不重复加载。
        - 选中其他技能：加载成功后替换当前技能。
        - 选择 None：移除当前技能。
        - 新技能加载失败：保留原技能并记录错误。
        - 记录选择结果，包括“不使用技能”。
        - 实际加载或移除时，记录对应事件。
        """
        raise NotImplementedError

    def get_instructions(self, session: SkillSession) -> str | None:
        """返回当前技能正文；无技能时返回 None。

        每轮模型调用前获取，用于构造运行时上下文。
        不向永久 base prompt 或原始聊天历史反复追加。
        """
        raise NotImplementedError

    def close_session(self, session: SkillSession, step: int) -> None:
        """结束当前问题，移除技能并记录事件。

        主 Agent 应在 finally 中调用，避免状态跨问题泄漏。
        """
        raise NotImplementedError
