from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.experimental.subagents import SubAgents

from synagent.analogues import AnalogueSearch
from synagent.chemspace import Chemspace
from synagent.corrector import Corrector
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
            "When the user asks to validate a route: call validate_route with route_json='from_message' — "
            "the tool reads the route directly from the conversation, you do NOT need to copy any SMILES or JSON. "
            "Report the ValidationReport results and STOP. Do not attempt to fix anything. "
            "When the user explicitly says 'fix' or 'correct': call fix_building_blocks() first, "
            "then call fix_step(step=N) for each failed step — both tools read the ValidationReport "
            "automatically, no SMILES copying needed. "
            "For other tasks, use the appropriate capability or sub-agent."
        ),
        capabilities=[
            AnalogueSearch(),
            # Chemspace(),
            SynthesisValidation(),
            Corrector(),
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
