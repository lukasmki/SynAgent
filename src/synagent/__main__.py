import logfire as lf
import typer
import uvicorn
from dotenv import load_dotenv
from pydantic_ai import Agent

from synagent.chemspace import Chemspace
from synagent.scoring import Scoring
from synagent.validation import SynthesisValidation

load_dotenv()

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

    agent = Agent(
        model,
        capabilities=[
            Chemspace(),
            SynthesisValidation(),
            Scoring(),
            # CodeMode(tools={"code_mode": True}),
        ],
    )
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
