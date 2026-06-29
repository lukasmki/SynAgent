import asyncio

import logfire as lf
import typer
import uvicorn
from dotenv import load_dotenv

from synagent.interface import interface
from synagent.synagent import get_agent

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

    agent = get_agent(model)
    uvicorn.run(agent.to_web(), host=host, port=port)


@app.command(name="cli")
def cli(model: str = "google:gemini-3-flash-preview", logfire: bool = False):
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()
    asyncio.run(interface(model))


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        # default
        cli()


def main():
    app()


if __name__ == "__main__":
    main()
