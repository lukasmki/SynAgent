import asyncio
import json

from dotenv import load_dotenv
from pydantic_ai import (
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
)
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from synagent.synagent import get_agent

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
    text_buffer: list[str] = []
    thinking_buffer: list[str] = []
    spinner_active = True

    console.print()

    spinner = console.status("[dim]Thinking…[/dim]", spinner="dots")
    spinner.start()

    def stop_spinner() -> None:
        nonlocal spinner_active
        if spinner_active:
            spinner.stop()
            spinner_active = False

    async with agent.run_stream_events(
        user_input, message_history=message_history
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
                content = (
                    str(event.content) if event.content is not None else "(no result)"
                )
                display = content[:500] + ("…" if len(content) > 500 else "")
                console.print(
                    Panel(
                        f"[dim]{display}[/dim]",
                        title="[bold green]Result[/bold green]",
                        border_style="green dim",
                    )
                )
            elif isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, TextPartDelta):
                    stop_spinner()
                    text_buffer.append(event.delta.content_delta)
                elif isinstance(event.delta, ThinkingPartDelta):
                    stop_spinner()
                    thinking_buffer.append(event.delta.content_delta)

            elif isinstance(event, AgentRunResultEvent):
                message_history.extend(event.result.new_messages())

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


async def interface(model: str) -> None:
    agent = get_agent(model)
    message_history: list = []
    turn: int = 0

    console.print(
        Panel(
            f"[bold green]Retrosynthesis assistant[/bold green]\n"
            f"[dim]Model:[/dim] [cyan]{model}[/cyan]\n\n"
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
    asyncio.run(interface("google:gemini-3-flash-preview"))
