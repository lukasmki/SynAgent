from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset

from synagent.corrector._toolset import CorrectorToolset

_FIX_TRIGGERS = {"fix", "correct", "repair", "search alternative", "alternative building block"}

# All corrector-owned tools
_ALL_CORRECTOR_TOOLS = {
    "fix_step", "fix_building_blocks", "apply_fixes", "search_step_building_blocks",
    "fix_smarts", "extract_template_from_reaction", "fix_template", "fix_smiles",
}

# Corrector tools hidden outside fix mode (gated until user asks to fix)
_GATED_TOOLS = _ALL_CORRECTOR_TOOLS


@dataclass
class Corrector(AbstractCapability[AgentDepsT]):
    id: str | None = "corrector"
    description: str | None = "Fix a specific failed step only when the user explicitly asks to fix or correct it."
    defer_loading: bool = False

    def get_instructions(self) -> str:
        return (
            "When the user asks to fix a failed route:\n"
            "1. Call fix_building_blocks() once.\n"
            "2. Call fix_step(step=N) once for each failed step — one call per step, no repeats.\n"
            "3. Call apply_fixes() once — applies all fixes and re-validates.\n"
            "4. Report the new ValidationReport and STOP.\n"
            "If that report still has failures on the same steps, do one more round: "
            "fix_step(N) once per still-failing step, then apply_fixes() once. Then STOP regardless.\n"
            "Do NOT call retro_search, save_record, search_step_building_blocks, "
            "search_building_blocks, or score_molecules unless the user explicitly asks for them. "
            "Do NOT call apply_fixes more than once per round. "
            "Never copy or retype SMILES yourself."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return CorrectorToolset()

    async def prepare_tools(
        self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        # Collect last 3 user messages — current message may not yet be in ctx.messages
        # so scanning a window catches "fix" from a message one turn ago
        recent: list[str] = []
        for msg in reversed(ctx.messages):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        recent.append(str(part.content).lower())
                        break
            if len(recent) >= 3:
                break

        in_fix_mode = any(
            word in msg for msg in recent for word in _FIX_TRIGGERS
        )

        if in_fix_mode:
            # Show all tools — corrector tools + all other capabilities' tools pass through
            return tool_defs

        # Outside fix mode: hide corrector tools, pass everything else through
        return [td for td in tool_defs if td.name not in _GATED_TOOLS]
