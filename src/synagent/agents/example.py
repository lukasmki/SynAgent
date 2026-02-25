import random

from pydantic_ai import Agent

SYSTEM_PROMPT = """
You are a helpful AI agent.
""".strip()

# Setup the agent
agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    output_type=str,
)


# Define a tool
@agent.tool_plain
def roll_dice() -> str:
    """Roll a six-sided die and return the result."""
    return str(random.randint(1, 6))
