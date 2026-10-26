"""Run SynAgent's `pipeline` workflow over every unique target in a SynLlama CSV.

The raw SynLlama output holds one row per (target, sampling_params) sample, so the
1000 targets appear ~10 times each; only the deduplicated `smiles` column is used
here. Each target goes through the full pipeline -- generate, validate, then
correct until the route validates or `--max-iter` is spent.

Two artifacts come out of a run:

* the results JSONL (`--out`): one line per target with the final `ValidationReport`
  and whether it validated. This is the authoritative target -> outcome record.
* the trace (`--trace-path`): every model call, tool call, and run, ready for
  `synagent trace export`. Runs from concurrent targets interleave in one file and
  separate again by `run_id`; a run is tied back to its target through the target
  SMILES carried in the generation run's prompt.

Both are append-only, so an interrupted run resumes with `--resume` (the default)
instead of paying for completed targets twice.

    uv run python data/synagent-generate.py --limit 5      # smoke test
    uv run python data/synagent-generate.py --workers 8    # the full 1000
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl
import typer
from dotenv import load_dotenv
from pydantic_ai import Agent
from tqdm import tqdm

from synagent.synagent import get_agent
from synagent.tracing import current_writer, resolve_trace_path
from synagent.validation import ValidationReport
from synagent.workflows import get_workflow

DATA = Path(__file__).parent
ROOT = DATA.parent

# Keys live in the repo-root .env; resolve it explicitly so the script behaves the
# same whether it is launched from the repo root or from data/.
load_dotenv(ROOT / ".env")

DEFAULT_MODEL = "google:gemini-3-flash-preview"

REQUEST = """\
Target molecule (SMILES): {smiles}

Propose a synthesis route to this exact target from commercially available building
blocks. Every step must use a reaction template that genuinely transforms its stated
reactants into its stated product, and the product of the final step must be the
target molecule."""


def unique_targets(csv: Path) -> list[str]:
    """Deduplicated target SMILES, in the order they first appear in the CSV."""
    df = pl.read_csv(csv.resolve())
    return df["smiles"].unique(maintain_order=True).to_list()


def completed_targets(out: Path, retry_failed: bool) -> set[str]:
    """Targets already in the results file that should not be run again.

    Records append, so a target rerun after a failure has several lines; the last
    one wins, matching how the file is meant to be read back.
    """
    if not out.exists():
        return set()
    status: dict[str, str] = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:  # a line torn by an earlier hard kill
            continue
        target = record.get("target")
        if target is not None:
            status[target] = record.get("status", "error")
    if retry_failed:
        return {t for t, s in status.items() if s == "ok"}
    return set(status)


class ResultWriter:
    """Appends one JSON line per target, flushed so a kill keeps finished work."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def run_target(
    agent: Agent[None, str],
    workflow: Any,
    index: int,
    smiles: str,
    max_iter: int,
    model: str,
) -> dict[str, Any]:
    """Run one target through the pipeline, turning any failure into a record.

    A single target must never take the batch down, so every exception -- API
    errors, output-validation failures, refusals -- is captured and reported.
    """
    started = perf_counter()
    record: dict[str, Any] = {
        "index": index,
        "target": smiles,
        "model": model,
        "ts": datetime.now(UTC).isoformat(),
    }
    try:
        report: ValidationReport = workflow(
            agent, REQUEST.format(smiles=smiles), max_iter=max_iter
        )
    except Exception as error:  # deliberate per-target isolation
        record |= {
            "status": "error",
            "valid": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    else:
        valid = report.all_building_blocks_valid and report.all_reactions_passed
        record |= {
            "status": "ok",
            "valid": valid,
            "all_building_blocks_valid": report.all_building_blocks_valid,
            "all_reactions_passed": report.all_reactions_passed,
            "n_reactions": len(report.reactions),
            "n_building_blocks": len(report.building_blocks),
            "report": report.model_dump(mode="json"),
            "error": None,
        }
    record["duration_s"] = perf_counter() - started
    return record


def main(
    csv: Path = typer.Option(
        DATA / "synllama-raw-output.csv", help="SynLlama raw output CSV."
    ),
    out: Path = typer.Option(
        DATA / "synagent-pipeline-results.jsonl", help="Results JSONL, appended to."
    ),
    model: str = typer.Option(DEFAULT_MODEL, help="LLM model identifier."),
    workers: int = typer.Option(4, help="Targets to run concurrently."),
    max_iter: int = typer.Option(4, help="Correction rounds per target."),
    limit: int | None = typer.Option(None, help="Only run the first N targets."),
    resume: bool = typer.Option(True, help="Skip targets already in the results file."),
    retry_failed: bool = typer.Option(
        False, help="With --resume, rerun targets whose last attempt errored."
    ),
    trace: bool = typer.Option(True, help="Write a JSONL trace for SFT export."),
    trace_path: Path | None = typer.Option(
        None, help="Trace file or directory (default .synagent/traces/)."
    ),
):
    """Run the full SynAgent pipeline on every unique target in the CSV."""
    targets = unique_targets(csv)
    if limit is not None:
        targets = targets[:limit]

    skip = completed_targets(out, retry_failed) if resume else set()
    pending = [(i, smi) for i, smi in enumerate(targets) if smi not in skip]
    if skip:
        print(
            f"Skipping {len(targets) - len(pending)} already-run targets",
            file=sys.stderr,
        )
    if not pending:
        print("Nothing to do.", file=sys.stderr)
        return

    path = (
        resolve_trace_path(trace_path or ROOT / ".synagent" / "traces")
        if trace
        else None
    )
    if path is not None:
        print(f"Tracing to {path}", file=sys.stderr)

    # One agent shared by every worker: the analogue databases are ~70MB of
    # in-memory fingerprints, so a per-worker agent would multiply that.
    agent = get_agent(model, trace=path)
    workflow = get_workflow("pipeline")
    writer = ResultWriter(out)

    counts = {"valid": 0, "invalid": 0, "error": 0}
    progress = tqdm(total=len(pending), desc="targets", unit="mol")
    # `run_sync` needs a thread without a running loop, which pool workers are;
    # each gets its own event loop the first time it runs a target.
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures: list[Future[dict[str, Any]]] = [
            pool.submit(run_target, agent, workflow, i, smi, max_iter, model)
            for i, smi in pending
        ]
        for future in as_completed(futures):
            record = future.result()
            writer.write(record)
            if record["status"] == "error":
                counts["error"] += 1
            else:
                counts["valid" if record["valid"] else "invalid"] += 1
            progress.update(1)
            progress.set_postfix(counts)
    except KeyboardInterrupt:
        # Completed targets are already on disk; --resume picks up the rest.
        print("\nInterrupted -- rerun with --resume to continue.", file=sys.stderr)
        pool.shutdown(wait=False, cancel_futures=True)
    finally:
        pool.shutdown(wait=True)
        progress.close()
        writer.close()
        if (tracer := current_writer()) is not None:
            tracer.close()

    done = sum(counts.values())
    print(f"\nRan {done}/{len(pending)} targets -> {out}", file=sys.stderr)
    for key, value in counts.items():
        share = value / done * 100 if done else 0.0
        print(f"{key}: {value}/{done}, {share:3.2f}%", file=sys.stderr)
    if path is not None:
        print(
            f"\nExport a fine-tuning set with:\n"
            f"  synagent trace export '{path}' -o dataset.jsonl --only-successful",
            file=sys.stderr,
        )


if __name__ == "__main__":
    typer.run(main)
