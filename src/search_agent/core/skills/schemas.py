from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SkillMetadata(BaseModel):
    """来自 SKILL.md 的 YAML frontmatter。

    发现阶段只向 Agent 提供这些信息，不提供技能正文。
    """

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1024)


class SkillInfo(BaseModel):
    """一个可用技能的目录信息，不包含完整指令。"""

    metadata: SkillMetadata
    directory: Path


class LoadedSkill(BaseModel):
    """运行时加载的技能及其完整 Markdown 指令。"""

    info: SkillInfo
    instructions: str = Field(min_length=1)


class SkillSearchContext(BaseModel):
    """自动选择技能所需的搜索状态摘要。

    由主 Agent 从自己的结构化状态生成。
    不存放完整聊天历史，也不复制整个文档集合。
    """

    question: str
    active_subqueries: list[str] = Field(default_factory=list)
    candidate_answer: str | None = None
    visited_doc_ids: list[str] = Field(default_factory=list)
    evidence_summary: str = ""
    last_query: str | None = None
    last_result_count: int | None = Field(default=None, ge=0)
    consecutive_failed_searches: int = Field(default=0, ge=0)


class SkillSelection(BaseModel):
    """自动选择结果。

    skill_name 为 None 表示当前不需要技能。
    """

    skill_name: str | None = None
    reason: str


class SkillSession(BaseModel):
    """单个问题的技能状态；每个问题独立创建。"""

    question_id: str
    active_skill: LoadedSkill | None = None


class SkillEvent(BaseModel):
    """交给主 Agent 统一写入 trajectory，并供界面展示。"""

    question_id: str
    step: int = Field(ge=0)
    event: Literal["selected", "loaded", "unloaded", "error"]
    skill_name: str | None = None
    reason: str = ""
