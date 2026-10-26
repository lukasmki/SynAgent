"""Full trace logging of prompts, responses, thoughts, and tool calls.

Records are written as JSONL, one `model_call` per model request/response pair --
the unit a supervised fine-tuning run consumes. `_export.py` converts a trace into
a trainer-ready dataset; the on-disk format stays lossless and pydantic-ai-native
so format changes never cost re-collection.
"""

from pathlib import Path

from synagent.tracing._capability import TraceLog
from synagent.tracing._export import export_sft
from synagent.tracing._models import (
    SCHEMA_VERSION,
    LabelRecord,
    ModelCallRecord,
    RunRecord,
    ToolResultRecord,
)
from synagent.tracing._writer import (
    TraceWriter,
    current_writer,
    default_trace_path,
    resolve_trace_path,
    trace_label,
)

__all__ = [
    "SCHEMA_VERSION",
    "LabelRecord",
    "ModelCallRecord",
    "RunRecord",
    "ToolResultRecord",
    "TraceLog",
    "TraceWriter",
    "build_tracing_capabilities",
    "current_writer",
    "default_trace_path",
    "export_sft",
    "resolve_trace_path",
    "trace_label",
]


def build_tracing_capabilities(trace: Path | None) -> list[TraceLog]:
    """Capabilities to splice into an agent, or `[]` when tracing is off."""
    if trace is None:
        return []
    return [TraceLog(writer=TraceWriter(trace))]
