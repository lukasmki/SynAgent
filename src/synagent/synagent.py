from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai_harness.subagents import SubAgent, SubAgents

from synagent.analogues import AnalogueSearch
from synagent.chemspace import Chemspace
from synagent.generation import Generation
from synagent.scoring import Scoring
from synagent.storage import Storage
from synagent.validation import SynthesisValidation


def get_agent(model: Model[Any] | KnownModelName | str) -> Agent[None, str]:
    validator = Agent(
        model,
        name="validator",
        description="Uses validation tools for validating synthesis routes.",
        instructions="You are a validation sub-agent.",
        capabilities=[SynthesisValidation()],
    )
    retriever = Agent(
        model,
        name="retriever",
        description="Can access the local building block and reactions database as well as the Chemspace API.",
        instructions="You are a molecule/reaction analogue retriever sub-agent.",
        capabilities=[AnalogueSearch(), Chemspace()],
    )
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
            Generation(),
            SynthesisValidation(),
            Scoring(),
            Storage(),
            SubAgents(
                agents=[
                    SubAgent(validator),
                    SubAgent(retriever),
                ],
            ),
        ],
    )
    return agent
