import asyncio
import sys
from pathlib import Path

import logfire as lf
import typer
import uvicorn
from dotenv import load_dotenv

from synagent.interface import interface
from synagent.spec import DEFAULT_SPEC, SCHEMA_PATH, write_schema
from synagent.synagent import get_agent
from synagent.workflows import get_workflow

load_dotenv()

app = typer.Typer(help="SynAgent")

MODEL_OPTION = typer.Option(
    None, help="LLM model identifier. Overrides the model set in the spec."
)
SPEC_OPTION = typer.Option(DEFAULT_SPEC, help="Agent spec file (YAML or JSON).")


@app.command(name="serve")
def serve(
    model: str | None = MODEL_OPTION,
    spec: Path = SPEC_OPTION,
    host: str = typer.Option("localhost", help="Host address to bind the server to."),
    port: int = typer.Option(8000, help="Port number to listen on."),
    logfire: bool = typer.Option(False, help="Enable Logfire monitoring and tracing."),
):
    """Start the HTTP web server."""
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()

    agent = get_agent(model, spec)
    uvicorn.run(agent.to_web(), host=host, port=port)


@app.command(name="cli")
def cli(
    model: str | None = MODEL_OPTION,
    spec: Path = SPEC_OPTION,
    logfire: bool = typer.Option(False, help="Enable Logfire monitoring and tracing."),
):
    """Start an interactive REPL with streaming tool calls."""
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()
    asyncio.run(interface(model, spec))


@app.command(name="run")
def run(
    name: str,
    request: str,
    model: str | None = MODEL_OPTION,
    spec: Path = SPEC_OPTION,
    logfire: bool = typer.Option(False, help="Enable Logfire monitoring and tracing."),
):
    """Run a request through a predefined agent workflow."""
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()
    agent = get_agent(model, spec)
    workflow = get_workflow(name)
    result = workflow(agent, request)
    print(result.model_dump_json(), file=sys.stdout)


@app.command(name="schema")
def schema(
    path: Path = typer.Argument(SCHEMA_PATH, help="Where to write the schema."),
):
    """Write the JSON schema for agent spec files, for editor validation."""
    print(write_schema(path), file=sys.stdout)


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    """Run the interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        asyncio.run(interface())


def main():
    app()


if __name__ == "__main__":
    main()
