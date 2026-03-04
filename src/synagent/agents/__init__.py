from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

from .example import agent as example_agent
from .validation import agent as validation_agent

load_dotenv()

DEFAULT_MODEL = GoogleModel(
    model_name="gemini-3-flash-preview",
    provider="google-gla",
    settings=GoogleModelSettings(
        google_thinking_config={"thinking_budget": 0},  # disable thinking
    ),
)

AGENTS = {"example": example_agent, "validation": validation_agent}


def get_agent(name: str) -> Agent:
    agent = AGENTS.get(name)
    if agent is None:
        raise ValueError(f"No agent with name {name}")
    if agent.model is None:
        agent.model = DEFAULT_MODEL
    return agent
