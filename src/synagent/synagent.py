from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.experimental.subagents import SubAgents

from synagent.analogues import AnalogueSearch
from synagent.chemspace import Chemspace
from synagent.corrector import Corrector
from synagent.retrosynthesis import Retrosynthesis
from synagent.scoring import Scoring
from synagent.storage import Storage
from synagent.validation import SynthesisValidation


def get_agent(model_name: str) -> Agent[None, str]:
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        settings=ModelSettings(thinking=False, extra_body={"think": False}),
    )

    agent = Agent(
        model,
        instructions=(
            "You are SynAgent, a synthesis planning assistant. "
            "VALIDATION: when asked to validate, call validate_route once, then report the COMPLETE "
            "ValidationReport — every building block (full SMILES, is_valid), every reaction step "
            "(number, template, reactants, product, actual_products, status, failure_mode), and "
            "suggested_fixes. Never summarize or omit fields. Then STOP — do not call any other tool. "
            "FIXING: when the user explicitly says 'fix' or 'correct', call fix_building_blocks() once, "
            "then fix_step(N) once per failed step, then apply_fixes() once. Report and STOP. "
            "RETROSYNTHESIS: only call retro_search when the user explicitly asks to redesign or find a new route. "
            "STORAGE: only call save_record or get_record when the user explicitly asks to save or retrieve. "
            "Never use a tool unless the user explicitly asked for that action. "
            "Never copy or retype SMILES or JSON yourself."
        ),
        capabilities=[
            AnalogueSearch(),
            # Chemspace(),
            SynthesisValidation(),
            Corrector(),
            Retrosynthesis(),
            Scoring(),
            Storage(),
            SubAgents(
                agents={
                    "analogue": Agent(
                        model,
                        description="Can access the local building block and reactions database as well as the Chemspace API.",
                        instructions="You are a molecule/reaction analogue sub-agent.",
                        capabilities=[AnalogueSearch(), Chemspace()],
                    ),
                    "worker": Agent(
                        model,
                        description="General purpose sub-agent.",
                        instructions="You are a sub-agent.",
                        capabilities=[
                            Chemspace(),
                            SynthesisValidation(),
                            AnalogueSearch(),
                        ],
                    ),
                },
                shared_capabilities=[],
            ),
            # CodeMode(tools={"code_mode": True}),
        ],
    )
    return agent
