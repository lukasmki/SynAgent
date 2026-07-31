from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset

from synagent.validation._toolset import SynthesisValidationToolset

_RETRO_TRIGGERS = {"reverse", "retro", "retrosynthesis", "backward", "backwards"}
_RETRO_TOOLS = {"reverse_reaction"}
# Always hidden — prevents the agent from delegating or loading capabilities during validation
_ALWAYS_HIDE = {"delegate_task", "load_capability"}
# Individual step tools are hidden during normal validation — validate_route handles everything
_STEP_TOOLS = {"validate_smiles", "validate_reaction_smarts", "create_report"}


@dataclass
class SynthesisValidation(AbstractCapability[AgentDepsT]):
    id = "synthesis-validation"
    description = "Use for synthesis path validation."
    defer_loading = False

    def get_instructions(self) -> str:
        return (
            "When the user asks to validate a route, call validate_route once with route_json='from_message'. "
            "The tool automatically extracts the route — do NOT copy or retype any SMILES or JSON. "
            "After validate_route returns, report the COMPLETE ValidationReport — show EVERY field: "
            "all_building_blocks_valid, all_reactions_passed, every building block with its full SMILES and is_valid, "
            "every reaction step with its reaction_number, reaction_template, reactant_smiles, expected_product, "
            "actual_products, status, failure_mode, and suggested_fix. Do NOT summarize or omit any field. "
            "Then STOP IMMEDIATELY. "
            "Do NOT call any other tool. Do NOT call fix_step, fix_building_blocks, apply_fixes, "
            "retro_search, save_record, search_building_blocks, or anything else. "
            "Validation only — fixing is a separate action the user must explicitly request."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return SynthesisValidationToolset()

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

        hide = set(_ALWAYS_HIDE) | _STEP_TOOLS
        if not any(word in last_user_msg for word in _RETRO_TRIGGERS):
            hide.add("reverse_reaction")

        return [td for td in tool_defs if td.name not in hide]
