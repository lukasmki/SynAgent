import logfire as lf
import typer
import uvicorn
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.subagents import SubAgents

from synagent.chemspace import Chemspace
from synagent.scoring import Scoring
from synagent.validation import SynthesisValidation

load_dotenv()


def get_agent(model: str) -> Agent[None, str]:
    agent = Agent(
        model,
        instructions=(
            "You are SynAgent, a synthesis planning assistant. "
            "Assist the user in planning and verification of synthesis routes. "
            "Utilize sub-agents to complete complex tasks in parallel."
        ),
        capabilities=[
            Chemspace(),
            SynthesisValidation(),
            Scoring(),
            SubAgents(
                agents={
                    "validator": Agent(
                        model,
                        instructions="You are a validation sub-agent.",
                        capabilities=[SynthesisValidation()],
                    ),
                    "chemspace": Agent(
                        model,
                        instructions="You are a Chemspace sub-agent.",
                        capabilities=[Chemspace()],
                    ),
                    "worker": Agent(
                        model,
                        instructions="You are a sub-agent.",
                        capabilities=[Chemspace(), SynthesisValidation()],
                    ),
                },
                shared_capabilities=[
                    Scoring(),
                ],
            ),
            # CodeMode(tools={"code_mode": True}),
        ],
    )
    return agent


app = typer.Typer()


@app.command(name="serve")
def serve(
    model: str = "google:gemini-3-flash-preview",
    host: str = "localhost",
    port: int = 8000,
    logfire: bool = False,
):
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()

    agent = get_agent(model)
    uvicorn.run(agent.to_web(), host=host, port=port)


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        # default
        serve()


def main():
    app()


if __name__ == "__main__":
    main()
