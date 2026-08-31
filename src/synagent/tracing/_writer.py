"""JSONL trace sink.

One file per process, date-partitioned, append-only. Every record carries a
`run_id`, so concurrent runs (the web server serves several at once) interleave
safely in a single file and separate again with a one-line `jq` filter.
"""

from __future__ import annotations

import os
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_core import to_json

from synagent.tracing._models import ErrorInfo, LabelRecord, _Envelope

_DEFAULT_ROOT = Path.cwd() / ".synagent" / "traces"

_current_writer: TraceWriter | None = None


def default_trace_path(root: Path | None = None) -> Path:
    """`.synagent/traces/YYYY-MM-DD/<start-ts>-<pid>.jsonl`."""
    now = datetime.now(UTC)
    root = root or _DEFAULT_ROOT
    return root / now.strftime("%Y-%m-%d") / f"{now:%H%M%S}-{os.getpid()}.jsonl"


def resolve_trace_path(value: Path | None) -> Path:
    """Turn a `--trace` option into a concrete file path.

    A directory (existing, or any path without a `.jsonl` suffix) gets an
    auto-named file inside it; anything else is used verbatim.
    """
    if value is None:
        return default_trace_path()
    if value.is_dir() or value.suffix != ".jsonl":
        return default_trace_path(root=value)
    return value


def error_info(error: BaseException) -> ErrorInfo:
    return ErrorInfo(
        type=type(error).__name__,
        message=str(error),
        traceback="".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    )


class TraceWriter:
    """Appends trace records to a JSONL file.

    Serialization goes through `pydantic_core.to_json(..., fallback=str)` so a
    non-serializable tool result (an RDKit mol, an FPSim2 array) degrades to its
    string form instead of raising in the middle of a run.
    """

    def __init__(self, path: Path, *, register: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        if register:
            global _current_writer
            _current_writer = self

    def write(self, record: _Envelope) -> None:
        line = to_json(record, fallback=str).decode()
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def label(self, run_id: str | None, **fields: Any) -> None:
        """Append ground truth for a run, joinable by `run_id`."""
        self.write(
            LabelRecord(ts=datetime.now(UTC), run_id=run_id, **fields)  # type: ignore[arg-type]
        )

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def current_writer() -> TraceWriter | None:
    """The process-wide writer, or `None` when tracing is off."""
    return _current_writer


def trace_label(run_id: str | None, **fields: Any) -> None:
    """Record ground truth for a run. No-ops when tracing is disabled."""
    writer = _current_writer
    if writer is not None:
        writer.label(run_id, **fields)
