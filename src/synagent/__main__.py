import asyncio
import sys
from glob import glob
from pathlib import Path

import logfire as lf
import typer
import uvicorn
from dotenv import load_dotenv

from synagent.interface import interface
from synagent.synagent import get_agent
from synagent.tracing import export_sft, resolve_trace_path
from synagent.workflows import get_workflow

load_dotenv()

app = typer.Typer(help="SynAgent")
trace_app = typer.Typer(help="Work with trace files.")
app.add_typer(trace_app, name="trace")

_MODEL_OPTION = typer.Option(
    "google:gemini-3-flash-preview", help="LLM model identifier."
)
_LOGFIRE_OPTION = typer.Option(False, help="Enable Logfire monitoring and tracing.")
_TRACE_OPTION = typer.Option(
    False, "--trace", help="Write a full JSONL trace under .synagent/traces/."
)
_TRACE_PATH_OPTION = typer.Option(
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
    model: str = _MODEL_OPTION,
    host: str = typer.Option("localhost", help="Host address to bind the server to."),
    port: int = typer.Option(8000, help="Port number to listen on."),
    logfire: bool = _LOGFIRE_OPTION,
    trace: bool = _TRACE_OPTION,
    trace_path: Path | None = _TRACE_PATH_OPTION,
):
    """Start the HTTP web server."""
    path = _setup_observability(logfire, trace, trace_path)
    agent = get_agent(model, trace=path)
    uvicorn.run(agent.to_web(), host=host, port=port)


@app.command(name="cli")
def cli(
    model: str = _MODEL_OPTION,
    logfire: bool = _LOGFIRE_OPTION,
    trace: bool = _TRACE_OPTION,
    trace_path: Path | None = _TRACE_PATH_OPTION,
):
    """Start an interactive REPL with streaming tool calls."""
    path = _setup_observability(logfire, trace, trace_path)
    asyncio.run(interface(model, trace=path))


@app.command(name="run")
def run(
    name: str,
    request: str,
    model: str = _MODEL_OPTION,
    logfire: bool = _LOGFIRE_OPTION,
    trace: bool = _TRACE_OPTION,
    trace_path: Path | None = _TRACE_PATH_OPTION,
):
    """Run a request through a predefined agent workflow."""
    path = _setup_observability(logfire, trace, trace_path)
    agent = get_agent(model, trace=path)
    workflow = get_workflow(name)
    result = workflow(agent, request)
    print(result.model_dump_json(), file=sys.stdout)


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
        asyncio.run(interface("google:gemini-3-flash-preview"))


def main():
    app()


if __name__ == "__main__":
    main()
