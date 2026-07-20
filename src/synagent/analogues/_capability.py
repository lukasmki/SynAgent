from pydantic_ai import AgentToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from synagent.analogues._toolset import AnalogueSearchToolset

_SEARCH_TRIGGERS = {"search", "find", "similar", "analogue", "analog", "building block", "template"}
_ANALOGUE_TOOLS = {"search_building_blocks", "search_templates", "search_building_blocks_by_template"}


class AnalogueSearch(AbstractCapability[AgentDepsT]):
    id = "analogue-search"
    description = "Use for finding reaction templates and building blocks."
    defer_loading = False

    def get_instructions(self) -> str:
        return "Use the analogue searching capability to find building blocks that fit reaction templates."

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return AnalogueSearchToolset()

    async def prepare_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        last_user_msg = ""
        for msg in reversed(ctx.messages):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        last_user_msg = str(part.content).lower()
                        break
                if last_user_msg:
                    break

        if any(word in last_user_msg for word in _SEARCH_TRIGGERS):
            return tool_defs
        return [td for td in tool_defs if td.name not in _ANALOGUE_TOOLS]
