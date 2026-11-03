import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import (
    AgentRunResult,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.messages import (
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

from synagent.spec import DEFAULT_SPEC
from synagent.synagent import USAGE_LIMITS, get_agent

console = Console()


def _print_help() -> None:
    console.print(
        Panel(
            "[bold]/help[/bold]   show this message\n"
            "[bold]/clear[/bold]  reset conversation history\n"
            "[bold]exit[/bold]    quit  (also: quit, /exit, Ctrl-C)",
            title="[dim]Commands[/dim]",
            border_style="dim",
            padding=(0, 2),
        )
    )


async def process_turn(agent, user_input: str, message_history: list) -> None:
    console.print()
    result = await stream_turn(agent, user_input, message_history)
    message_history.extend(result.new_messages())


async def stream_turn(
    agent,
    user_input: str,
    message_history: list,
    *,
    output_type: Any = None,
    console: Console = console,
    spinner: bool = True,
) -> AgentRunResult:
    """Run one turn, printing tool calls, thinking and the reply as they arrive.

    `message_history` is read, not extended. `output_type` overrides the agent's
    for this turn. Pass `spinner=False` where rich can't animate, such as a
    notebook.
    """
    text_buffer: list[str] = []
    thinking_buffer: list[str] = []
    result: AgentRunResult | None = None
    status = console.status("[dim]Thinking…[/dim]", spinner="dots") if spinner else None
    if status is not None:
        status.start()

    def stop_spinner() -> None:
        nonlocal status
        if status is not None:
            status.stop()
            status = None

    try:
        async with agent.run_stream_events(
            user_input,
            output_type=output_type,
            message_history=message_history,
            usage_limits=USAGE_LIMITS,
        ) as events:
            async for event in events:
                if isinstance(event, PartStartEvent):
                    if isinstance(event.part, TextPart) and event.part.content:
                        stop_spinner()
                        text_buffer.append(event.part.content)
                    elif isinstance(event.part, ThinkingPart) and event.part.content:
                        stop_spinner()
                        thinking_buffer.append(event.part.content)

                elif isinstance(event, FunctionToolCallEvent):
                    stop_spinner()
                    args = event.part.args_as_dict()
                    console.print(
                        Panel(
                            f"[dim]{json.dumps(args, indent=2)}[/dim]",
                            title=f"[bold yellow]Tool: {event.part.tool_name}[/bold yellow]",
                            border_style="yellow",
                        )
                    )

                elif isinstance(event, FunctionToolResultEvent):
                    stop_spinner()
                    part = event.part
                    if isinstance(part, ToolReturnPart):
                        content = part.model_response_str()
                        title = f"[bold green]Result: {part.tool_name}[/bold green]"
                        border_style = "green dim"
                    else:
                        content = part.model_response()
                        title = f"[bold red]Retry: {part.tool_name}[/bold red]"
                        border_style = "red dim"
                    display = content[:500] + ("…" if len(content) > 500 else "")
                    console.print(
                        Panel(
                            f"[dim]{escape(display)}[/dim]",
                            title=title,
                            border_style=border_style,
                        )
                    )
                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, TextPartDelta):
                        stop_spinner()
                        text_buffer.append(event.delta.content_delta)
                    elif isinstance(event.delta, ThinkingPartDelta):
                        stop_spinner()
                        # Signature-only deltas carry no content.
                        if event.delta.content_delta:
                            thinking_buffer.append(event.delta.content_delta)

                elif isinstance(event, AgentRunResultEvent):
                    result = event.result

    finally:
        stop_spinner()

    if thinking_buffer:
        console.print(
            Panel(
                f"[dim italic]{''.join(thinking_buffer)}[/dim italic]",
                title="[dim]Thinking[/dim]",
                border_style="dim",
            )
        )

    if text_buffer:
        console.print(Markdown("".join(text_buffer)))

    if result is None:
        raise RuntimeError("the event stream ended without a result")
    return result


async def interface(
    model: str | None = None,
    spec: Path | str = DEFAULT_SPEC,
    trace: Path | None = None,
) -> None:
    agent = get_agent(model, spec, trace=trace)
    message_history: list = []
    turn: int = 0

    console.print(
        Panel(
            f"[bold green]Retrosynthesis assistant[/bold green]\n"
            f"[dim]Model:[/dim] [cyan]{getattr(agent.model, 'model_name', agent.model)}[/cyan]\n\n"
            f"[dim]/help[/dim]  commands   "
            f"[dim]/clear[/dim]  new conversation   "
            f"[dim]exit[/dim]  quit",
            title="[bold green]SynAgent[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )
    console.print()

    while True:
        try:
            user_input = Prompt.ask(f"[bold cyan][{turn + 1}] >[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            break

        stripped = user_input.strip()
        if not stripped or stripped.lower() in ("exit", "quit", "/exit"):
            break

        if stripped.startswith("/"):
            if stripped == "/help":
                _print_help()
            elif stripped == "/clear":
                message_history.clear()
                console.print("[dim]Conversation cleared.[/dim]")
            else:
                console.print(
                    f"[dim]Unknown command: {stripped}. Type /help for help.[/dim]"
                )
            continue

        turn += 1
        try:
            await process_turn(agent, stripped, message_history)
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")

    console.print("\n[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(interface())
