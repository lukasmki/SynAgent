from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

from .chemspace import agent as chemspace_agent
from .validation import agent as validation_agent
from .optimization import agent as optimization_agent
from .corrector import agent as corrector_agent
from .master import agent as master_agent

load_dotenv()

DEFAULT_MODEL = GoogleModel(
    model_name="gemini-3.1-flash-lite",
    provider="google-gla",
    settings=GoogleModelSettings(
        google_thinking_config={"thinking_budget": 1024},
    ),
)

AGENTS = {"chemspace": chemspace_agent, "validation": validation_agent, "optimization": optimization_agent, "corrector": corrector_agent, "master":master_agent}


def get_agent(name: str) -> Agent:
    agent = AGENTS.get(name)
    if agent is None:
        raise ValueError(f"No agent with name {name}")
    if agent.model is None:
        agent.model = DEFAULT_MODEL
    return agent
