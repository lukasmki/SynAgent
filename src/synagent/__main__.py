import asyncio
import sys
from glob import glob
from pathlib import Path

import logfire as lf
import typer
import uvicorn
from dotenv import load_dotenv

from synagent.interface import interface
from synagent.spec import DEFAULT_SPEC, SCHEMA_PATH, write_schema
from synagent.synagent import get_agent
from synagent.tracing import export_sft, resolve_trace_path
from synagent.workflows import get_workflow

load_dotenv()

app = typer.Typer(help="SynAgent")
trace_app = typer.Typer(help="Work with trace files.")
app.add_typer(trace_app, name="trace")

MODEL_OPTION = typer.Option(
    None, help="LLM model identifier. Overrides the model set in the spec."
)
SPEC_OPTION = typer.Option(DEFAULT_SPEC, help="Agent spec file (YAML or JSON).")
LOGFIRE_OPTION = typer.Option(False, help="Enable Logfire monitoring and tracing.")
TRACE_OPTION = typer.Option(
    False, "--trace", help="Write a full JSONL trace under .synagent/traces/."
)
TRACE_PATH_OPTION = typer.Option(
    None, "--trace-path", help="Trace file or directory (implies --trace)."
)


def _setup_observability(
    logfire: bool, trace: bool, trace_path: Path | None
) -> Path | None:
    """Configure Logfire if asked, and resolve where traces go. `None` = no tracing."""
    if logfire:
        lf.configure()
        lf.instrument_pydantic_ai()

    if not trace and trace_path is None:
        return None
    path = resolve_trace_path(trace_path)
    print(f"Tracing to {path}", file=sys.stderr)
    return path


@app.command(name="serve")
def serve(
    model: str | None = MODEL_OPTION,
    spec: Path = SPEC_OPTION,
    host: str = typer.Option("localhost", help="Host address to bind the server to."),
    port: int = typer.Option(8000, help="Port number to listen on."),
    logfire: bool = LOGFIRE_OPTION,
    trace: bool = TRACE_OPTION,
    trace_path: Path | None = TRACE_PATH_OPTION,
):
    """Start the HTTP web server."""
    path = _setup_observability(logfire, trace, trace_path)
    agent = get_agent(model, spec, trace=path)
    uvicorn.run(agent.to_web(), host=host, port=port)


@app.command(name="cli")
def cli(
    model: str | None = MODEL_OPTION,
    spec: Path = SPEC_OPTION,
    logfire: bool = LOGFIRE_OPTION,
    trace: bool = TRACE_OPTION,
    trace_path: Path | None = TRACE_PATH_OPTION,
):
    """Start an interactive REPL with streaming tool calls."""
    path = _setup_observability(logfire, trace, trace_path)
    asyncio.run(interface(model, spec, trace=path))


@app.command(name="run")
def run(
    name: str,
    request: str,
    model: str | None = MODEL_OPTION,
    spec: Path = SPEC_OPTION,
    logfire: bool = LOGFIRE_OPTION,
    trace: bool = TRACE_OPTION,
    trace_path: Path | None = TRACE_PATH_OPTION,
):
    """Run a request through a predefined agent workflow."""
    path = _setup_observability(logfire, trace, trace_path)
    agent = get_agent(model, spec, trace=path)
    workflow = get_workflow(name)
    result = workflow(agent, request)
    print(result.model_dump_json(), file=sys.stdout)


@app.command(name="schema")
def schema(
    path: Path = typer.Argument(SCHEMA_PATH, help="Where to write the schema."),
):
    """Write the JSON schema for agent spec files, for editor validation."""
    print(write_schema(path), file=sys.stdout)


@trace_app.command(name="export")
def trace_export(
    patterns: list[str] = typer.Argument(
        ..., help="Trace files or globs, e.g. '.synagent/traces/**/*.jsonl'."
    ),
    out: Path = typer.Option(..., "-o", "--out", help="Destination JSONL dataset."),
    include_thinking: bool = typer.Option(
        False, help="Keep thinking content as `reasoning_content` on the target."
    ),
    only_successful: bool = typer.Option(
        False, help="Keep only runs that succeeded and were not labelled invalid."
    ),
    agent: str | None = typer.Option(None, help="Keep only this agent's calls."),
    min_steps: int = typer.Option(0, help="Drop runs shorter than this many steps."),
):
    """Convert traces into a chat-format supervised fine-tuning dataset."""
    paths = [Path(p) for pattern in patterns for p in glob(pattern, recursive=True)]
    if not paths:
        raise typer.BadParameter(f"No trace files matched: {', '.join(patterns)}")
    written = export_sft(
        paths,
        out,
        include_thinking=include_thinking,
        only_successful=only_successful,
        agent=agent,
        min_steps=min_steps,
    )
    print(
        f"Wrote {written} samples from {len(paths)} file(s) to {out}", file=sys.stderr
    )


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    """Run the interactive REPL when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        asyncio.run(interface())


def main():
    app()


if __name__ == "__main__":
    main()
