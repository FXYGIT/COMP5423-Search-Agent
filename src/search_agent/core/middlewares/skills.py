"""运行时技能目录和已选技能指令的请求级注入。"""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.runtime import Runtime

from search_agent.core.skills import SkillLoader
from search_agent.core.state import SearchState


class SkillMiddleware(AgentMiddleware[SearchState, None]):
    """向主模型提供技能目录，并按请求注入当前技能正文。"""

    state_schema = SearchState

    def __init__(self, loader: SkillLoader | None = None) -> None:
        self.loader = loader or SkillLoader()

    def before_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        """Keep discovery metadata out of state and let the model choose a skill."""
        return {}

    async def abefore_model(self, state: SearchState, runtime: Runtime[None]) -> dict[str, Any]:
        return self.before_model(state, runtime)

    def prepare_request(self, request: ModelRequest[None]) -> ModelRequest[None]:
        metadata = self._catalog()
        state = request.state or {}
        active = state.get("active_skill")
        instructions = state.get("skill_instructions", "")
        sections = [request.system_prompt or "", metadata]
        if active and instructions:
            sections.append(
                f"\n## Active skill: {active}\n"
                "Follow these instructions for the current search step only:\n\n"
                f"{instructions}"
            )
        return request.override(
            system_prompt="\n\n".join(section for section in sections if section)
        )

    def _catalog(self) -> str:
        skills = self.loader.discover()
        if not skills:
            return "## Available skills\nNo optional skills are currently installed."
        entries = "\n".join(f"- `{skill.name}`: {skill.description}" for skill in skills)
        return (
            "## Available skills\n"
            "Choose a skill by calling `load_skill` when it is useful; otherwise use no skill.\n"
            f"{entries}"
        )

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Inject a fresh request view; the base prompt remains unchanged."""
        return handler(self.prepare_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """异步模型请求使用同一请求级技能视图。"""
        return await handler(self.prepare_request(request))
