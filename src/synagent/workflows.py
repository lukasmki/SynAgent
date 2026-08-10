from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent

from synagent.models import SynLlamaFormat
from synagent.validation import ValidationReport

Workflow = Callable[..., BaseModel]
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


@workflow("generation")
def generation(agent: Agent[None, str], request: str) -> SynLlamaFormat:
    result = agent.run_sync(
        user_prompt=f"Request:\n{request}",
        instructions="Generate a synthesis path adhering to the request.",
        output_type=SynLlamaFormat,
    )
    synthesis = result.output
    return synthesis


@workflow("validation")
def validation(
    agent: Agent[None, str], synthesis: str | SynLlamaFormat
) -> ValidationReport:
    if isinstance(synthesis, str):
        synthesis = SynLlamaFormat.model_validate_json(synthesis)

    result = agent.run_sync(
        user_prompt=f"Synthesis path:\n{synthesis.model_dump_json()}",
        instructions=(
            "Validate the provided synthesis path. "
            "Return the validated pathway as your final result."
        ),
        output_type=ValidationReport,
    )
    report = result.output
    return report


@workflow("correction")
def correction(
    agent: Agent[None, str], report: str | ValidationReport
) -> ValidationReport:
    if isinstance(report, str):
        report = ValidationReport.model_validate_json(report)

    result = agent.run_sync(
        user_prompt=f"Synthesis Path:\n{report.model_dump_json()}",
        instructions=(
            "Correct the errors in the provided synthesis pathway. "
            "Return the updated pathway as your final result."
        ),
        output_type=ValidationReport,
    )
    report = result.output
    return report


@workflow("pipeline")
def pipeline(
    agent: Agent[None, str], request: str, max_iter: int = 4
) -> ValidationReport:
    synthesis = generation(agent, request)
    report = validation(agent, synthesis)
    valid = report.all_building_blocks_valid & report.all_reactions_passed
    iiter = 0
    while not (valid | (iiter >= max_iter)):
        report = correction(agent, report)
        valid = report.all_building_blocks_valid & report.all_reactions_passed
        iiter += 1
    return report
