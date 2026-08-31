"""Pydantic schemas for the four trace record types.

These models are the contract between the writer and the exporter: records are
written by dumping them and read back with `model_validate_json`. `schema_version`
is bumped whenever a field changes meaning, so old traces stay interpretable.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class ErrorInfo(BaseModel):
    """A captured exception."""

    type: str
    message: str
    traceback: str | None = None


class ModelInfo(BaseModel):
    """Which model produced a response, and under what settings."""

    name: str | None = None
    provider: str | None = None
    settings: dict[str, Any] | None = None


class ToolSchema(BaseModel):
    """A tool definition as the model saw it.

    Part of the prompt, so it must be captured per request: which tools are in
    scope changes as capabilities load and defer.
    """

    name: str
    description: str | None = None
    parameters_json_schema: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None
    kind: Literal["function", "output"] = "function"


class ModelCallInput(BaseModel):
    """Everything the model was given for one call."""

    messages: list[dict[str, Any]]
    instructions: str | None = None
    output_mode: str | None = None
    tools: list[ToolSchema] = Field(default_factory=list)


class _Envelope(BaseModel):
    """Fields common to every record."""

    schema_version: int = SCHEMA_VERSION
    ts: datetime
    run_id: str | None = None
    conversation_id: str | None = None
    agent: str | None = None
    parent_run_id: str | None = None
    parent_tool_call_id: str | None = None


class ModelCallRecord(_Envelope):
    """One model request/response pair -- the supervised fine-tuning unit."""

    record: Literal["model_call"] = "model_call"
    step: int = 0
    model: ModelInfo = Field(default_factory=ModelInfo)
    input: ModelCallInput
    output: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    latency_ms: float | None = None
    error: ErrorInfo | None = None


class ToolResultRecord(_Envelope):
    """One tool execution, joined to its `model_call` by `tool_call_id`.

    Holds the raw Python result, which the `ToolReturnPart` in the next step's
    messages does not -- that is the model-facing rendering.
    """

    record: Literal["tool_result"] = "tool_result"
    step: int = 0
    tool_name: str
    tool_call_id: str | None = None
    args: Any = None
    result: Any = None
    duration_ms: float | None = None
    retry: int = 0
    error: ErrorInfo | None = None
    phase: Literal["execute", "validate"] = "execute"


class RunRecord(_Envelope):
    """One agent run, written at completion. Carries the filtering counts."""

    record: Literal["run"] = "run"
    status: Literal["ok", "error"] = "ok"
    model: ModelInfo = Field(default_factory=ModelInfo)
    prompt: Any = None
    output: Any = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    duration_ms: float | None = None
    n_steps: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    n_retries: int = 0
    error: ErrorInfo | None = None


class LabelRecord(_Envelope):
    """Post-hoc ground truth for a run, appended by callers that know the outcome.

    Extra fields are allowed: the label vocabulary is caller-defined (e.g.
    `valid` and `iteration` from the `pipeline` workflow).
    """

    model_config = ConfigDict(extra="allow")

    record: Literal["label"] = "label"


TraceRecord = ModelCallRecord | ToolResultRecord | RunRecord | LabelRecord
