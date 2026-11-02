"""Python interface to SynAgent for notebooks and scripts.

    import synagent as sa

    agent = sa.SynAgent(spec="nogpu")
    chat = agent.chat()
    reply = await chat.ask("Is CCO a building block?")
    report = await agent.validate(sa.Route.from_steps([...]))

Every method that runs the agent is async, for notebooks, and has a blocking
`*_sync` twin for scripts.
"""

import asyncio
import threading
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from pydantic_ai import Agent, AgentRunResult, AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import RunUsage
from rich.console import Console

from synagent import workflows
from synagent.interface import stream_turn
from synagent.models import SynLlamaFormat
from synagent.spec import list_specs, resolve_spec
from synagent.synagent import get_agent
from synagent.tracing import resolve_trace_path
from synagent.validation import ValidationReport

Route = SynLlamaFormat
"A synthesis route: `Route.from_steps(...)` or `Route.from_json(...)`"

Report = ValidationReport
"The result of validating a route"


def specs() -> list[str]:
    """Names of the bundled specs, for `SynAgent(spec=...)`."""
    return list_specs()


@dataclass
class ToolCall:
    """One tool call the agent made, and what came back."""

    name: str
    args: dict[str, Any]
    id: str
    result: Any = None
    "The tool's return value, or the retry message when `retried`"
    retried: bool = False
    "The call failed validation or raised, and the model was asked to retry"


class ToolCalls(list[ToolCall]):
    """The tool calls of one reply, in order. Displays as a table in a notebook."""

    def _repr_html_(self) -> str:
        if not self:
            return "<p>No tool calls.</p>"
        rows = "".join(
            f"<tr><td>{i}</td><td>{escape(call.name)}</td>"
            f"<td><code>{escape(_short(_format_args(call.args)))}</code></td>"
            f"<td>{'retry: ' if call.retried else ''}"
            f"<code>{escape(_short(repr(call.result)))}</code></td></tr>"
            for i, call in enumerate(self, start=1)
        )
        return (
            "<table><tr><th>#</th><th>tool</th><th>args</th><th>result</th></tr>"
            f"{rows}</table>"
        )


def _format_args(args: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _short(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_calls(messages: Iterable[ModelMessage]) -> ToolCalls:
    """Pair each tool call with its return or retry by tool call id."""
    calls: dict[str, ToolCall] = {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                calls[part.tool_call_id] = ToolCall(
                    name=part.tool_name,
                    args=part.args_as_dict(),
                    id=part.tool_call_id,
                )
            elif isinstance(part, ToolReturnPart) and part.tool_call_id in calls:
                call = calls[part.tool_call_id]
                call.result = part.content
                call.retried = part.outcome == "retried"
    return ToolCalls(calls.values())


@dataclass(repr=False)
class Reply:
    """The agent's answer to one prompt. Displays as Markdown in a notebook."""

    text: str
    tool_calls: ToolCalls
    usage: RunUsage
    result: AgentRunResult[str]
    "The underlying pydantic-ai result"
    _streamed: bool = False
    "The answer was already printed while the agent worked"

    @classmethod
    def from_result(cls, result: AgentRunResult[str], *, streamed: bool) -> "Reply":
        return cls(
            text=result.output,
            tool_calls=_tool_calls(result.new_messages()),
            usage=result.usage,
            result=result,
            _streamed=streamed,
        )

    def show(self) -> None:
        """Display the answer as Markdown, even if it was already streamed."""
        from IPython.display import Markdown, display

        display(Markdown(self.text))

    def _ipython_display_(self) -> None:
        # A streamed answer is already in the cell output; don't show it twice.
        if not self._streamed:
            self.show()

    def __repr__(self) -> str:
        return (
            f"Reply({len(self.tool_calls)} tool calls, "
            f"{self.usage.requests} requests): {_short(self.text)!r}"
        )


class SynAgent:
    """SynAgent built from a spec, with chat and the `synagent run` workflows.

    `spec` is a bundled spec name (see `specs()`) or a spec file path, and `model`
    overrides the spec's model. `trace=True` writes a JSONL trace to the default
    location; a path writes it there. With `dotenv`, API keys are loaded from the
    nearest `.env` above the working directory.
    """

    def __init__(
        self,
        *,
        spec: str | Path = "synagent",
        model: Model[Any] | KnownModelName | str | None = None,
        trace: bool | str | Path = False,
        dotenv: bool = True,
    ) -> None:
        if dotenv:
            load_dotenv(find_dotenv(usecwd=True))
        self.spec = spec
        self.spec_path = resolve_spec(spec)
        self.trace_path: Path | None = None
        if trace is not False:
            self.trace_path = resolve_trace_path(None if trace is True else Path(trace))
        self.agent: Agent[None, str] = get_agent(
            model, self.spec_path, trace=self.trace_path
        )
        self.capabilities: list[str] = [
            c.name for c in AgentSpec.from_file(self.spec_path).capabilities
        ]

    def __repr__(self) -> str:
        model = self.agent.model
        name = (
            f"{model.system}:{model.model_name}" if isinstance(model, Model) else model
        )
        return (
            f"SynAgent(spec={str(self.spec)!r}, model={name!r})\n"
            f"  capabilities: {', '.join(self.capabilities) or 'none'}"
        )

    def chat(self) -> "Chat":
        """Start a conversation that remembers its history."""
        return Chat(self)

    async def ask(self, prompt: str, *, show_tools: bool = True) -> Reply:
        """Run one prompt with no history. See `Chat.ask` for `show_tools`."""
        return await self._turn(prompt, [], show_tools=show_tools)

    async def validate(self, route: Route | str) -> Report:
        """Check each step of `route` and its building blocks."""
        self._require("validate", "SynthesisValidation", "SubAgents")
        return await workflows.validation(self.agent, route)

    async def correct(self, report: Report | str) -> Report:
        """Fix the failed steps in `report` and return the updated report."""
        self._require("correct", "SynthesisValidation", "SubAgents")
        return await workflows.correction(self.agent, report)

    async def generate(self, request: str) -> Route:
        """Propose a route for a natural-language request with SynLlama."""
        self._require("generate", "Generation")
        return await workflows.generation(self.agent, request)

    async def pipeline(self, request: str, max_iter: int = 4) -> Report:
        """Generate a route, then validate and correct it until it passes."""
        self._require("pipeline", "Generation")
        self._require("pipeline", "SynthesisValidation", "SubAgents")
        return await workflows.pipeline(self.agent, request, max_iter=max_iter)

    def ask_sync(self, prompt: str, *, show_tools: bool = True) -> Reply:
        return _run_sync(self.ask(prompt, show_tools=show_tools), "ask")

    def validate_sync(self, route: Route | str) -> Report:
        return _run_sync(self.validate(route), "validate")

    def correct_sync(self, report: Report | str) -> Report:
        return _run_sync(self.correct(report), "correct")

    def generate_sync(self, request: str) -> Route:
        return _run_sync(self.generate(request), "generate")

    def pipeline_sync(self, request: str, max_iter: int = 4) -> Report:
        return _run_sync(self.pipeline(request, max_iter), "pipeline")

    async def _turn(
        self, prompt: str, history: list[ModelMessage], *, show_tools: bool
    ) -> Reply:
        if show_tools:
            # A fresh Console picks up the notebook, if there is one.
            result = await stream_turn(
                self.agent, prompt, history, console=Console(), spinner=False
            )
        else:
            result = await self.agent.run(prompt, message_history=history)
        return Reply.from_result(result, streamed=show_tools)

    def _require(self, method: str, *capabilities: str) -> None:
        """Fail before any model call when the spec can't run `method`."""
        if not any(c in self.capabilities for c in capabilities):
            raise UserError(
                f"{method}() needs the {' or '.join(capabilities)} capability, "
                f"which spec {str(self.spec)!r} doesn't load."
            )


class Chat:
    """A conversation with a `SynAgent`. Each `ask` sees the turns before it."""

    def __init__(self, agent: SynAgent) -> None:
        self.agent = agent
        self.history: list[ModelMessage] = []

    def __repr__(self) -> str:
        return f"Chat({len(self.history)} messages)"

    async def ask(self, prompt: str, *, show_tools: bool = True) -> Reply:
        """Send `prompt` and return the reply.

        With `show_tools`, tool calls, thinking and the answer are printed as the
        agent works, as in `synagent cli`. Otherwise nothing prints until the
        reply is displayed.
        """
        reply = await self.agent._turn(prompt, self.history, show_tools=show_tools)
        self.history.extend(reply.result.new_messages())
        return reply

    def ask_sync(self, prompt: str, *, show_tools: bool = True) -> Reply:
        return _run_sync(self.ask(prompt, show_tools=show_tools), "ask")

    def reset(self) -> None:
        """Forget the conversation, like `/clear` in the REPL."""
        self.history.clear()


_thread_loops = threading.local()


def _run_sync[T](coro: Coroutine[Any, Any, T], method: str) -> T:
    """Run `coro` to completion on this thread's event loop.

    The loop is kept open between calls: the model clients are cached, and their
    pooled connections stay bound to the loop that opened them.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()
        raise RuntimeError(
            "An event loop is already running, as in a notebook: "
            f"use `await` with `{method}()` instead of `{method}_sync()`."
        )
    loop = getattr(_thread_loops, "loop", None)
    if loop is None:
        loop = _thread_loops.loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)
