from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from synagent.models import SynLlamaFormat
from synagent.synagent import USAGE_LIMITS
from synagent.tracing import trace_label
from synagent.validation import ValidationReport

Workflow = Callable[..., Awaitable[BaseModel]]
WorkflowT = TypeVar("WorkflowT", bound=Workflow)

WORKFLOWS: dict[str, Workflow] = {}


def get_workflow(name: str) -> Workflow:
    try:
        return WORKFLOWS[name]
    except KeyError:
        known = ", ".join(sorted(WORKFLOWS)) or "none"
        raise KeyError(
            f"Unknown workflow {name!r}. Available workflows: {known}"
        ) from None


def workflow(name: str | None = None) -> Callable[[WorkflowT], WorkflowT]:
    """Register a workflow function under `name` (defaults to its own name)."""

    def decorator(fn: WorkflowT) -> WorkflowT:
        key = name or getattr(fn, "__name__", None)
        if not key:
            raise ValueError(f"Cannot infer a workflow name for {fn!r}; pass one.")
        if key in WORKFLOWS:
            raise ValueError(f"Workflow {key!r} is already registered.")
        WORKFLOWS[key] = fn
        return fn

    return decorator


async def _run_step(
    agent: Agent[None, str],
    *,
    user_prompt: str,
    instructions: str,
    output_type: type[Any],
) -> tuple[Any, str]:
    """Run one step and hand back its run id, so callers can label the trajectory."""
    result = await agent.run(
        user_prompt=user_prompt,
        instructions=instructions,
        output_type=output_type,
        usage_limits=USAGE_LIMITS,
    )
    return result.output, result.run_id


async def _generation(
    agent: Agent[None, str], request: str
) -> tuple[SynLlamaFormat, str]:
    return await _run_step(
        agent,
        user_prompt=f"Request:\n{request}",
        instructions="Generate a synthesis path adhering to the request.",
        output_type=SynLlamaFormat,
    )


async def _validation(
    agent: Agent[None, str], synthesis: str | SynLlamaFormat
) -> tuple[ValidationReport, str]:
    if isinstance(synthesis, str):
        synthesis = SynLlamaFormat.model_validate_json(synthesis)

    return await _run_step(
        agent,
        user_prompt=f"Synthesis path:\n{synthesis.model_dump_json()}",
        instructions=(
            "Validate the provided synthesis path. "
            "Return the validated pathway as your final result."
        ),
        output_type=ValidationReport,
    )


async def _correction(
    agent: Agent[None, str], report: str | ValidationReport
) -> tuple[ValidationReport, str]:
    if isinstance(report, str):
        report = ValidationReport.model_validate_json(report)

    return await _run_step(
        agent,
        user_prompt=f"Synthesis Path:\n{report.model_dump_json()}",
        instructions=(
            "Correct the errors in the provided synthesis pathway. "
            "Return the updated pathway as your final result."
        ),
        output_type=ValidationReport,
    )


@workflow("generation")
async def generation(agent: Agent[None, str], request: str) -> SynLlamaFormat:
    synthesis, _ = await _generation(agent, request)
    return synthesis


@workflow("validation")
async def validation(
    agent: Agent[None, str], synthesis: str | SynLlamaFormat
) -> ValidationReport:
    report, _ = await _validation(agent, synthesis)
    return report


@workflow("correction")
async def correction(
    agent: Agent[None, str], report: str | ValidationReport
) -> ValidationReport:
    corrected, _ = await _correction(agent, report)
    return corrected


@workflow("pipeline")
async def pipeline(
    agent: Agent[None, str], request: str, max_iter: int = 4
) -> ValidationReport:
    synthesis, _ = await _generation(agent, request)
    report, run_id = await _validation(agent, synthesis)
    valid = report.passed
    iiter = 0
    # `valid` is ground truth for the trajectory: recorded per run so a fine-tuning
    # set can be filtered down to the runs that actually produced a valid synthesis.
    trace_label(run_id, workflow="pipeline", valid=valid, iteration=iiter)
    while not valid and iiter < max_iter:
        report, run_id = await _correction(agent, report)
        valid = report.passed
        iiter += 1
        trace_label(run_id, workflow="pipeline", valid=valid, iteration=iiter)
    return report
