import asyncio

import logfire as lf
import typer
import uvicorn
from dotenv import load_dotenv

from synagent.interface import interface
from synagent.synagent import get_agent

load_dotenv()

app = typer.Typer(help="SynAgent")


@app.command(name="serve")
def serve(
    model: str = typer.Option(
        "google:gemini-3-flash-preview", help="LLM model identifier."
    ),
    host: str = typer.Option("localhost", help="Host address to bind the server to."),
    port: int = typer.Option(8000, help="Port number to listen on."),
    logfire: bool = typer.Option(False, help="Enable Logfire monitoring and tracing."),
):
    """Start the HTTP web server."""
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()

    agent = get_agent(model)
    uvicorn.run(agent.to_web(), host=host, port=port)


@app.command(name="cli")
def cli(
    model: str = typer.Option(
        "google:gemini-3-flash-preview", help="LLM model identifier."
    ),
    logfire: bool = typer.Option(False, help="Enable Logfire monitoring and tracing."),
):
    """Start an interactive REPL with streaming tool calls."""
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()
    asyncio.run(interface(model))


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    """Run the interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        asyncio.run(interface("google:gemini-3-flash-preview"))


def main():
    app()


if __name__ == "__main__":
    main()
