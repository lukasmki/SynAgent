from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from synagent.models import BuildingBlockResult, ReactionResult, SynLlamaReport, OptimizationReport
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
import re
from typing import Literal
from pydantic_ai.models.google import GoogleModel
import os
from dotenv import load_dotenv

load_dotenv()
OPTIMIZER_PROMPT = """
        You are a retrosynthesis optimizer.
        You receive a SynLlamaReport.
        Only optimize based on building block price and hazard information.
        Use DuckDuckGo search only to look up building block price and hazard information.
        Do not evaluate reaction yield.
        If a building block is invalid, flag it rather than searching deeply.
        Summarize evidence conservatively and return a structured OptimizationReport""".strip()

model = GoogleModel('gemini-3-flash-preview')
###################
# Search only helper #
###################

search_agent = Agent(
    model,
    tools = [duckduckgo_search_tool(max_results=5)],
    output_type = str,
    system_prompt=(
        "You are a chemical search assistant.\n"
        "Use DuckDuckGo search to retrieve concise public web evidence.\n"
        "Prefer supplier pages, catalog pages, SDS pages, and chemistry references.\n"
        "Do not invent facts.\n"
        "Return a short evidence summary only."
    )
)

# @search_agent.tool_plain
# def search_by_smiles(smiles: str):
    



###################
# Search optimizer #
###################

agent = Agent(
    model,
    system_prompt=OPTIMIZER_PROMPT,
    output_type=OptimizationReport,
)


###################
# Optimizer report (search) #
################### 

class PriceLookupResult(BaseModel):
    smiles: str
    query_used: str
    price_text: str | None = None
    estimated_price_usd: float | None = None
    currency: str | None = None
    vendor_hint: str | None = None
    evidence: str

class HazardLookupResult(BaseModel):
    smiles: str
    query_used: str
    hazard_flags: list[str] = []
    hazard_score: float = Field(default=0.5, ge=0, le=1)  # higher = safer
    evidence: str

class AvailabilityLookupResult(BaseModel):
    smiles: str
    query_used: str
    availability: Literal["available", "likely_available", "unclear", "not_found"]
    vendor_hints: list[str] = []
    evidence: str

###################
# route level #
###################

class RouteCostResult(BaseModel):
    per_block: list[PriceLookupResult]
    total_estimated_cost_usd: float | None = None
    missing_price_count: int
    cost_score: float = Field(ge=0, le=1)

class RouteSafetyResult(BaseModel):
    per_block: list[HazardLookupResult]
    average_hazard_score: float = Field(ge=0, le=1)
    flagged_blocks: list[str] = []
    safety_score: float = Field(ge=0, le=1)

class BuildingBlockEvaluation(BaseModel):
    smiles: str
    name: str | None = None
    valid: bool
    price: PriceLookupResult | None = None
    hazard: HazardLookupResult | None = None
    availability: AvailabilityLookupResult | None = None

class OptimizationScore(BaseModel):
    cost_score: float = Field(ge=0, le=1)
    safety_score: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)

class OptimizationReport(BaseModel):
    is_optimizable: bool
    summary: str
    target_molecule: str
    issues_found: list[str]
    building_block_evaluations: list[BuildingBlockEvaluation]
    route_cost: RouteCostResult
    route_safety: RouteSafetyResult
    score: OptimizationScore
    recommended_actions: list[str]

###################
# Words & price to look at #
###################

_PRICE_RE = re.compile(r"(?i)(?:\$|usd\s*)(\d+(?:\.\d+)?)")

_VENDOR_WORDS = [
    "sigma",
    "aldrich",
    "tc i",
    "combi-blocks",
    "oakwood",
    "abcr",
    "apollo",
    "fluorochem",
    "thermo",
    "fisher",
    "vwr",
    "cayman",
    "broadpharm",
    "molport",
]

_HAZARD_KEYWORDS: dict[str, float] = {
    "fatal": 0.05,
    "toxic": 0.15,
    "corrosive": 0.15,
    "carcinogenic": 0.05,
    "mutagen": 0.05,
    "flammable": 0.45,
    "combustible": 0.55,
    "irritant": 0.65,
    "harmful": 0.55,
    "oxidizer": 0.30,
    "explosive": 0.05,
    "danger": 0.25,
    "warning": 0.60,
}

###################
# search for price and harzard #
###################
def _extract_estimated_price(text:str) -> tuple[float | None, str | None]:
    match = _PRICE_RE.search(text)
    if not match:
        return None, None
    try:
        return float(match.group(1)), "USD"
    except ValueError:
        return None, None

def _extract_vendor_hint(text: str) -> str | None:
    lower = text.lower()
    for vendor in _VENDOR_WORDS:
        if vendor in lower:
            return vendor
    return None

def _score_hazard_from_text(text: str) -> tuple[list[str], float]:
    lower = text.lower()
    flags: list[str] = []
    scores: list[float] = []

    for word, score in _HAZARD_KEYWORDS.items():
        if word in lower:
            flags.append(word)
            scores.append(score)

    if not scores:
        return [], 0.8

    return flags, min(scores)

def _availability_from_text(text: str) -> tuple[Literal["available", "likely_available", "unclear", "not_found"], list[str]]:
    lower = text.lower()
    vendors = [vendor for vendor in _VENDOR_WORDS if vendor in lower]

    if any(term in lower for term in ["in stock", "add to cart", "buy now", "catalog number"]):
        return "available", vendors

    if vendors or any(term in lower for term in ["supplier", "catalog", "product page"]):
        return "likely_available", vendors

    if any(term in lower for term in ["not available", "discontinued", "out of stock"]):
        return "not_found", vendors

    return "unclear", vendors

def _cost_to_score(total_cost: float | None) -> float:
    if total_cost is None:
        return 0.5
    return max(0.0, min(1.0, 1 / (1 + total_cost / 100.0)))

def _combine_scores(cost_score: float, safety_score: float) -> float:
    return round((0.5 * cost_score) + (0.5 * safety_score), 3)

@agent.tool_plain
async def lookup_building_block_price(smiles: str, name: str | None = None) -> PriceLookupResult:
    query_term = name or smiles
    query = f'"{query_term}" chemical supplier price OR catalog'
    result = await search_agent.run(query)
    evidence = result.output

    price, currency = _extract_estimated_price(evidence)
    vendor = _extract_vendor_hint(evidence)

    return PriceLookupResult(
        smiles=smiles,
        query_used=query,
        price_text=(f"{currency} {price}" if price is not None and currency else None),
        estimated_price_usd=price,
        currency=currency,
        vendor_hint=vendor,
        evidence=evidence,
    )

@agent.tool_plain
async def lookup_building_block_hazard(smiles: str, name: str | None = None) -> HazardLookupResult:
    query_term = name or smiles
    query = f'"{query_term}" SDS OR hazard OR GHS'
    result = await search_agent.run(query)
    evidence = result.output

    flags, hazard_score = _score_hazard_from_text(evidence)

    return HazardLookupResult(
        smiles=smiles,
        query_used=query,
        hazard_flags=flags,
        hazard_score=hazard_score,
        evidence=evidence,
    )

@agent.tool_plain
async def lookup_building_block_availability(smiles: str, name: str | None = None) -> AvailabilityLookupResult:
    query_term = name or smiles
    query = f'"{query_term}" supplier OR catalog OR in stock'
    result = await search_agent.run(query)
    evidence = result.output

    availability, vendors = _availability_from_text(evidence)

    return AvailabilityLookupResult(
        smiles=smiles,
        query_used=query,
        availability=availability,
        vendor_hints=vendors,
        evidence=evidence,
    )

@agent.tool_plain
async def compute_route_cost(report: SynLlamaReport) -> RouteCostResult:
    per_block: list[PriceLookupResult] = []

    for bb in report.building_blocks:
        if not bb.is_valid:
            per_block.append(
                PriceLookupResult(
                    smiles=bb.smiles,
                    query_used="",
                    evidence="Building block marked invalid by validation agent.",
                )
            )
            continue

        per_block.append(await lookup_building_block_price(bb.smiles, None))

    prices = [item.estimated_price_usd for item in per_block if item.estimated_price_usd is not None]
    total_cost = sum(prices) if prices else None
    missing_price_count = sum(1 for item in per_block if item.estimated_price_usd is None)

    return RouteCostResult(
        per_block=per_block,
        total_estimated_cost_usd=total_cost,
        missing_price_count=missing_price_count,
        cost_score=_cost_to_score(total_cost),
    )
@agent.tool_plain
async def compute_route_safety(report: SynLlamaReport) -> RouteSafetyResult:
    per_block: list[HazardLookupResult] = []

    for bb in report.building_blocks:
        if not bb.is_valid:
            per_block.append(
                HazardLookupResult(
                    smiles=bb.smiles,
                    query_used="",
                    hazard_flags=["invalid_building_block"],
                    hazard_score=0.0,
                    evidence="Building block marked invalid by validation agent.",
                )
            )
            continue

        per_block.append(await lookup_building_block_hazard(bb.smiles, bb.name))

    scores = [item.hazard_score for item in per_block]
    average_hazard_score = sum(scores) / len(scores) if scores else 0.0
    flagged_blocks = [
        item.smiles for item in per_block
        if item.hazard_score < 0.5 or len(item.hazard_flags) > 0
    ]

    return RouteSafetyResult(
        per_block=per_block,
        average_hazard_score=round(average_hazard_score, 4),
        flagged_blocks=flagged_blocks,
        safety_score=round(average_hazard_score, 4),
    )

@agent.tool_plain
async def evaluate_building_blocks(report: SynLlamaReport) -> list[BuildingBlockEvaluation]:
    evaluations: list[BuildingBlockEvaluation] = []

    for bb in report.building_blocks:
        if not bb.is_valid:
            evaluations.append(
                BuildingBlockEvaluation(
                    smiles=bb.smiles,
                    name=bb.name,
                    valid=False,
                    price=PriceLookupResult(
                        smiles=bb.smiles,
                        query_used="",
                        evidence="Building block marked invalid by validation agent.",
                    ),
                    hazard=HazardLookupResult(
                        smiles=bb.smiles,
                        query_used="",
                        hazard_flags=["invalid_building_block"],
                        hazard_score=0.0,
                        evidence="Building block marked invalid by validation agent.",
                    ),
                    availability=AvailabilityLookupResult(
                        smiles=bb.smiles,
                        query_used="",
                        availability="not_found",
                        evidence="Building block marked invalid by validation agent.",
                    ),
                )
            )
            continue

        price = await lookup_building_block_price(bb.smiles, bb.name)
        hazard = await lookup_building_block_hazard(bb.smiles, bb.name)
        availability = await lookup_building_block_availability(bb.smiles, bb.name)

        evaluations.append(
            BuildingBlockEvaluation(
                smiles=bb.smiles,
                name=bb.name,
                valid=True,
                price=price,
                hazard=hazard,
                availability=availability,
            )
        )

    return evaluations

@agent.tool_plain
async def optimize_route(report: SynLlamaReport) -> OptimizationReport:
    evaluations = await evaluate_building_blocks(report)
    route_cost = await compute_route_cost(report)
    route_safety = await compute_route_safety(report)

    issues_found: list[str] = []
    recommended_actions: list[str] = []

    if not report.all_building_blocks_valid:
        issues_found.append("One or more building blocks are invalid.")
        recommended_actions.append("Fix invalid building block SMILES before further optimization.")

    if route_cost.missing_price_count > 0:
        issues_found.append(
            f"Price evidence was incomplete for {route_cost.missing_price_count} building block(s)."
        )

    if route_safety.flagged_blocks:
        issues_found.append(
            f"Hazard flags were found for {len(route_safety.flagged_blocks)} building block(s)."
        )
        recommended_actions.append("Review SDS and handling precautions for flagged building blocks.")

    expensive_blocks = [
        ev.smiles
        for ev in evaluations
        if ev.price is not None and ev.price.estimated_price_usd is not None and ev.price.estimated_price_usd > 100
    ]
    if expensive_blocks:
        issues_found.append(f"Potentially expensive building blocks detected: {', '.join(expensive_blocks)}.")
        recommended_actions.append("Consider replacing the most expensive building blocks with cheaper alternatives.")

    scarce_blocks = [
        ev.smiles
        for ev in evaluations
        if ev.availability is not None and ev.availability.availability == "not_found"
    ]
    if scarce_blocks:
        issues_found.append(f"Potential sourcing problems detected: {', '.join(scarce_blocks)}.")
        recommended_actions.append("Prioritize routes using commercially available building blocks.")

    if not recommended_actions:
        recommended_actions.append("Current building block set appears acceptable for this early screening stage.")

    cost_score = route_cost.cost_score
    safety_score = route_safety.safety_score
    overall_score = _combine_scores(cost_score, safety_score)

    score = OptimizationScore(
        cost_score=round(cost_score, 4),
        safety_score=round(safety_score, 4),
        overall_score=overall_score,
    )

    is_optimizable = report.all_building_blocks_valid

    summary_parts = [
        f"Target molecule: {report.target_molecule}.",
        f"Assessed {len(report.building_blocks)} building block(s).",
        f"Cost score: {score.cost_score:.2f}.",
        f"Safety score: {score.safety_score:.2f}.",
        f"Overall score: {score.overall_score:.2f}.",
    ]
    if expensive_blocks:
        summary_parts.append("Some building blocks may be expensive.")
    if route_safety.flagged_blocks:
        summary_parts.append("Some building blocks may carry hazard concerns.")

    return OptimizationReport(
        is_optimizable=is_optimizable,
        summary=" ".join(summary_parts),
        target_molecule=report.target_molecule,
        issues_found=issues_found,
        building_block_evaluations=evaluations,
        route_cost=route_cost,
        route_safety=route_safety,
        score=score,
        recommended_actions=recommended_actions,
    )