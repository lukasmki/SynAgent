from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from synagent.models import SynLlamaFormat
from synagent.validation import ValidationReport


def get_workflow(name) -> Callable[[Agent[None, str], Any], BaseModel]:
    workflows = {
        "generation": generation,
        "validation": validation,
        "correction": correction,
        "pipeline": pipeline,
    }
    return workflows[name]


def generation(agent: Agent[None, str], request: str) -> SynLlamaFormat:
    result = agent.run_sync(
        user_prompt=f"Request:\n{request}",
        instructions="Generate a synthesis path adhering to the request.",
        output_type=SynLlamaFormat,
    )
    synthesis = result.output
    return synthesis


def validation(
    agent: Agent[None, str], synthesis: str | SynLlamaFormat
) -> ValidationReport:
    if isinstance(synthesis, str):
        synthesis = SynLlamaFormat.model_validate_json(synthesis)

    result = agent.run_sync(
        user_prompt=f"Synthesis path:\n{synthesis.model_dump_json()}",
        instructions="Validate the provided synthesis path.",
        output_type=ValidationReport,
    )
    report = result.output
    return report


def correction(
    agent: Agent[None, str], report: str | ValidationReport
) -> ValidationReport:
    if isinstance(report, str):
        report = ValidationReport.model_validate_json(report)

    result = agent.run_sync(
        user_prompt=f"Synthesis Path:\n{report.model_dump_json()}",
        instructions="Correct the errors in the provided synthesis pathway.",
        output_type=ValidationReport,
    )
    report = result.output
    return report


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
