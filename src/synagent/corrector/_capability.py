from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset

from synagent.corrector._toolset import CorrectorToolset

_FIX_TRIGGERS = {"fix", "correct", "repair"}

# Tools shown only during fix mode
_FIX_TOOLS = {"fix_step", "fix_building_blocks", "fix_smarts", "extract_template_from_reaction",
              "fix_template", "fix_building_block", "fix_smiles", "fix_target", "validate_products"}

# Fix tools hidden outside fix mode
_GATED_TOOLS = {"fix_step", "fix_building_blocks", "fix_smarts", "extract_template_from_reaction",
                "fix_template", "fix_building_block", "fix_smiles", "fix_target"}


@dataclass
class Corrector(AbstractCapability[AgentDepsT]):
    id = "corrector"
    description = "Fix a specific failed step only when the user explicitly asks to fix or correct it."
    defer_loading = False

    def get_instructions(self) -> str:
        return (
            "When the user asks to fix a failed route, use these two tools — no SMILES copying needed:\n"
            "1. Call fix_building_blocks() — fixes all invalid building block SMILES from the last ValidationReport.\n"
            "2. For each failed reaction step, call fix_step(step=N) — reads the ValidationReport automatically "
            "and runs the correct fix chain for that step's failure_mode.\n"
            "Report what each tool returns. "
            "If fix_step returns new_reactants, those are alternative precursors to use for that step. "
            "For a hard-to-synthesize target, call fix_target with the target SMILES from the ValidationReport. "
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
            # Whitelist: only fix tools + validate_products visible — nothing else
            return [td for td in tool_defs if td.name in _FIX_TOOLS]

        # Outside fix mode: hide fix tools, pass everything else through
        return [td for td in tool_defs if td.name not in _GATED_TOOLS]
