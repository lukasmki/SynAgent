from pydantic_ai import Agent
from pydantic_ai_harness.experimental.subagents import SubAgents

from synagent.analogues import AnalogueSearch
from synagent.chemspace import Chemspace
from synagent.scoring import Scoring
from synagent.storage import Storage
from synagent.validation import SynthesisValidation


def get_agent(model: str) -> Agent[None, str]:
    agent = Agent(
        model,
        instructions=(
            "You are SynAgent, a synthesis planning assistant. "
            "Assist the user in planning and verification of synthesis routes. "
            "Utilize sub-agents to complete complex tasks in parallel."
        ),
        capabilities=[
            AnalogueSearch(),
            Chemspace(),
            SynthesisValidation(),
            Scoring(),
            Storage(),
            SubAgents(
                agents={
                    "validator": Agent(
                        model,
                        description="Uses RDKit tools for validating synthesis routes.",
                        instructions="You are a validation sub-agent.",
                        capabilities=[SynthesisValidation()],
                    ),
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
