"""Capability that records a full trace of every model call and tool execution.

Registered once on the agent, it covers `run_stream_events`, `run_sync`, and the
web server alike, because every path goes through the same lifecycle hooks.

Every hook here is strictly pass-through: it returns what the wrapped handler
returned and re-raises what it raised, so enabling tracing can never change how
the agent behaves.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.tools import AgentDepsT
from pydantic_core import from_json, to_json

from synagent.tracing._models import (
    ModelCallInput,
    ModelCallRecord,
    ModelInfo,
    RunRecord,
    ToolResultRecord,
    ToolSchema,
)
from synagent.tracing._writer import TraceWriter, error_info

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.run import AgentRunResult
    from pydantic_ai.tools import RunContext, ToolDefinition

# Set while a tool executes, so a sub-agent run started inside that tool can name
# its parent. The harness awaits `agent.run(...)` inside `wrap_tool_execute`
# (pydantic_ai_harness/experimental/subagents/_toolset.py), so the child sees this;
# concurrently delegated sub-agents each get their own context copy.
_parent_link: ContextVar[tuple[str | None, str | None] | None] = ContextVar(
    "synagent_trace_parent", default=None
)


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Best-effort JSON dict for dataclasses, pydantic models, and TypedDicts."""
    if value is None:
        return None
    dumped = from_json(to_json(value, fallback=str))
    return dumped if isinstance(dumped, dict) else {"value": dumped}


def _instructions_of(messages: list[ModelMessage]) -> str | None:
    """The instructions actually rendered for this request.

    Capabilities inject instructions dynamically, so this varies step to step and
    cannot be recovered from the agent definition after the fact.
    """
    for message in reversed(messages):
        instructions = getattr(message, "instructions", None)
        if instructions is not None:
            return instructions
    return None


def _tool_schemas(params: Any) -> list[ToolSchema]:
    def convert(tool: ToolDefinition, kind: str) -> ToolSchema:
        return ToolSchema(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=dict(tool.parameters_json_schema),
            strict=tool.strict,
            kind=kind,  # type: ignore[arg-type]
        )

    return [
        *(convert(t, "function") for t in params.function_tools),
        *(convert(t, "output") for t in params.output_tools),
    ]


@dataclass
class _RunState:
    started: float
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None
    n_steps: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    n_retries: int = 0


@dataclass
class TraceLog(AbstractCapability[AgentDepsT]):
    """Writes `model_call`, `tool_result`, and `run` records to a `TraceWriter`."""

    writer: TraceWriter
    id: str | None = field(default="trace-log", kw_only=True)
    description: str | None = field(
        default="Records a full trace of prompts, responses, thoughts, and tool calls.",
        kw_only=True,
    )
    _runs: dict[str, _RunState] = field(default_factory=dict, repr=False, init=False)

    # --- helpers ---

    def _state(self, ctx: RunContext[Any]) -> _RunState:
        key = ctx.run_id or ""
        state = self._runs.get(key)
        if state is None:
            parent = _parent_link.get() or (None, None)
            state = _RunState(
                started=perf_counter(),
                parent_run_id=parent[0],
                parent_tool_call_id=parent[1],
            )
            self._runs[key] = state
        return state

    def _envelope(self, ctx: RunContext[Any]) -> dict[str, Any]:
        state = self._state(ctx)
        state.n_steps = max(state.n_steps, ctx.run_step)
        return {
            "ts": datetime.now(UTC),
            "run_id": ctx.run_id,
            "conversation_id": ctx.conversation_id,
            "agent": ctx.agent.name if ctx.agent is not None else None,
            "parent_run_id": state.parent_run_id,
            "parent_tool_call_id": state.parent_tool_call_id,
        }

    # --- run lifecycle ---

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        # Establishes the run's start time and parent link before anything else fires.
        self._state(ctx)

    async def after_run(
        self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        self._write_run(ctx, result=result, error=None)
        return result

    async def on_run_error(
        self, ctx: RunContext[AgentDepsT], *, error: BaseException
    ) -> AgentRunResult[Any]:
        self._write_run(ctx, result=None, error=error)
        raise error

    def _write_run(
        self,
        ctx: RunContext[Any],
        *,
        result: AgentRunResult[Any] | None,
        error: BaseException | None,
    ) -> None:
        state = self._state(ctx)
        messages: list[dict[str, Any]] = []
        if result is not None:
            messages = ModelMessagesTypeAdapter.dump_python(
                result.all_messages(), mode="json"
            )
        self.writer.write(
            RunRecord(
                **self._envelope(ctx),
                status="error" if error is not None else "ok",
                model=ModelInfo(
                    name=getattr(ctx.model, "model_name", None),
                    provider=getattr(ctx.model, "system", None),
                ),
                prompt=ctx.prompt,
                output=result.output if result is not None else None,
                messages=messages,
                usage=_as_dict(result.usage if result is not None else ctx.usage),
                duration_ms=(perf_counter() - state.started) * 1000,
                n_steps=state.n_steps,
                n_tool_calls=state.n_tool_calls,
                n_tool_errors=state.n_tool_errors,
                n_retries=state.n_retries,
                error=error_info(error) if error is not None else None,
            )
        )
        self._runs.pop(ctx.run_id or "", None)

    # --- model requests: one record per call, the fine-tuning unit ---

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: Any,
    ) -> ModelResponse:
        start = perf_counter()
        try:
            response = await handler(request_context)
        except Exception as error:
            self._write_model_call(ctx, request_context, None, start, error)
            raise
        self._write_model_call(ctx, request_context, response, start, None)
        return response

    def _write_model_call(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
        response: ModelResponse | None,
        start: float,
        error: Exception | None,
    ) -> None:
        params = request_context.model_request_parameters
        # Serialize the response through the same adapter as the messages so the
        # exporter sees one consistent shape.
        output = (
            ModelMessagesTypeAdapter.dump_python([response], mode="json")[0]
            if response is not None
            else None
        )
        self.writer.write(
            ModelCallRecord(
                **self._envelope(ctx),
                step=ctx.run_step,
                model=ModelInfo(
                    name=getattr(request_context.model, "model_name", None),
                    provider=getattr(request_context.model, "system", None),
                    settings=_as_dict(
                        ctx.model_settings or request_context.model_settings
                    ),
                ),
                input=ModelCallInput(
                    messages=ModelMessagesTypeAdapter.dump_python(
                        request_context.messages, mode="json"
                    ),
                    instructions=_instructions_of(list(request_context.messages)),
                    output_mode=params.output_mode,
                    tools=_tool_schemas(params),
                ),
                output=output,
                usage=_as_dict(response.usage) if response is not None else None,
                latency_ms=(perf_counter() - start) * 1000,
                error=error_info(error) if error is not None else None,
            )
        )

    # --- tool execution ---

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        handler: Any,
    ) -> Any:
        state = self._state(ctx)
        state.n_tool_calls += 1
        start = perf_counter()
        # Marks this tool as the parent of any sub-agent run started inside it.
        token = _parent_link.set((ctx.run_id, call.tool_call_id))
        try:
            result = await handler(args)
        except Exception as error:
            state.n_tool_errors += 1
            self._write_tool(ctx, call, args, None, start, error, "execute")
            raise
        finally:
            _parent_link.reset(token)
        self._write_tool(ctx, call, args, result, start, None, "execute")
        return result

    async def on_tool_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        error: Exception,
    ) -> Any:
        # Malformed tool calls are valuable negatives, so they are recorded too.
        state = self._state(ctx)
        state.n_tool_errors += 1
        state.n_retries += 1
        self._write_tool(ctx, call, args, None, perf_counter(), error, "validate")
        raise error

    def _write_tool(
        self,
        ctx: RunContext[Any],
        call: ToolCallPart,
        args: Any,
        result: Any,
        start: float,
        error: Exception | None,
        phase: str,
    ) -> None:
        self.writer.write(
            ToolResultRecord(
                **self._envelope(ctx),
                step=ctx.run_step,
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
                args=args,
                result=result,
                duration_ms=(perf_counter() - start) * 1000,
                retry=ctx.retry,
                error=error_info(error) if error is not None else None,
                phase=phase,  # type: ignore[arg-type]
            )
        )
