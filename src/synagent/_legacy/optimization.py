from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import prod
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

model = GoogleModel("gemini-3-flash-preview")

OPTIMIZER_PROMPT = """
You are a route hazard scoring assistant.

Your job is to:
1. read the hazard information for each compound,
2. compute compound-level hazard scores,
3. compute the overall route hazard score,
4. explain the result clearly.

Use the provided hazard scoring functions directly.
""".strip()

agent = Agent(
    model,
    output_type=str,
    system_prompt=OPTIMIZER_PROMPT,
)


HAZARD_WEIGHTS: dict[str, float] = {
    # Acute toxicity
    "H300": 1.00,  # Fatal if swallowed
    "H310": 1.00,  # Fatal in contact with skin
    "H330": 1.00,  # Fatal if inhaled
    "H301": 0.82,  # Toxic if swallowed
    "H311": 0.82,  # Toxic in contact with skin
    "H331": 0.82,  # Toxic if inhaled
    "H302": 0.58,  # Harmful if swallowed
    "H312": 0.58,  # Harmful in contact with skin
    "H332": 0.58,  # Harmful if inhaled
    # Corrosion / irritation / sensitization
    "H314": 0.72,  # Causes severe skin burns and eye damage
    "H315": 0.30,  # Causes skin irritation
    "H318": 0.62,  # Causes serious eye damage
    "H319": 0.25,  # Causes serious eye irritation
    "H317": 0.36,  # May cause an allergic skin reaction
    "H334": 0.80,  # May cause allergy or asthma symptoms or breathing difficulties if inhaled
    # CMR
    "H340": 0.93,  # May cause genetic defects
    "H341": 0.68,  # Suspected of causing genetic defects
    "H350": 0.95,  # May cause cancer
    "H351": 0.72,  # Suspected of causing cancer
    "H360": 0.93,  # May damage fertility or the unborn child
    "H361": 0.70,  # Suspected of damaging fertility or the unborn child
    "H362": 0.45,  # May cause harm to breast-fed children
    # STOT / aspiration
    "H370": 0.88,  # Causes damage to organs
    "H371": 0.63,  # May cause damage to organs
    "H335": 0.35,  # May cause respiratory irritation
    "H336": 0.35,  # May cause drowsiness or dizziness
    "H372": 0.90,  # Causes damage to organs through prolonged/repeated exposure
    "H373": 0.66,  # May cause damage to organs through prolonged/repeated exposure
    "H304": 0.83,  # May be fatal if swallowed and enters airways
    "H305": 0.50,  # May be harmful if swallowed and enters airways
    # Explosives / flammability / reactivity
    "H200": 1.00,  # Unstable explosive
    "H201": 0.98,  # Explosive; mass explosion hazard
    "H202": 0.96,  # Explosive; severe projection hazard
    "H203": 0.92,  # Explosive; fire, blast or projection hazard
    "H204": 0.72,  # Fire or projection hazard
    "H205": 0.76,  # May mass explode in fire
    "H220": 0.72,  # Extremely flammable gas
    "H221": 0.48,  # Flammable gas
    "H224": 0.70,  # Extremely flammable liquid and vapor
    "H225": 0.55,  # Highly flammable liquid and vapor
    "H226": 0.38,  # Flammable liquid and vapor
    "H227": 0.22,  # Combustible liquid
    "H228": 0.45,  # Flammable solid
    "H240": 0.95,  # Heating may cause an explosion
    "H241": 0.85,  # Heating may cause a fire or explosion
    "H242": 0.68,  # Heating may cause a fire
    "H250": 0.90,  # Catches fire spontaneously if exposed to air
    "H251": 0.76,  # Self-heating; may catch fire
    "H252": 0.45,  # Self-heating in large quantities
    "H260": 0.86,  # In contact with water releases flammable gases which may ignite spontaneously
    "H261": 0.68,  # In contact with water releases flammable gases
    "H270": 0.78,  # May cause or intensify fire; oxidizer
    "H271": 0.90,  # May cause fire or explosion; strong oxidizer
    "H272": 0.46,  # May intensify fire; oxidizer
    "H230": 0.95,  # May react explosively even in absence of air
    "H231": 0.88,  # May react explosively even in absence of air at elevated pressure and/or temperature
    # Environmental
    "H400": 0.40,
    "H410": 0.46,
    "H411": 0.33,
    "H412": 0.20,
    "H413": 0.10,
}


RED_FLAG_CODES = {
    "H200",
    "H201",
    "H202",
    "H203",
    "H205",
    "H240",
    "H271",
    "H250",
    "H300",
    "H310",
    "H330",
}


@dataclass
class CompoundHazard:
    name: str
    hazard_codes: list[str]
    matched_weights: list[float] = field(default_factory=list)
    compound_hazard: float = 0.0
    red_flag: bool = False


def compound_hazard_score(
    hazard_codes: Iterable[str],
) -> tuple[float, list[float], bool]:
    """
    Combine hazard codes into one compound-level hazard score using noisy-OR.

    Returns:
        (
            compound_score in [0, 1],
            matched weight list,
            red_flag boolean,
        )
    """
    weights: list[float] = []
    red_flag = False
    codes: list[str] = []

    for raw in hazard_codes:
        code = raw.strip().upper()
        codes.append(code)

        if code in HAZARD_WEIGHTS:
            weights.append(HAZARD_WEIGHTS[code])

        if code in RED_FLAG_CODES:
            red_flag = True

    # Extra red flag: carcinogen + fatal acute toxicity
    has_carc1 = "H350" in codes
    has_fatal_acute = any(code in codes for code in ("H300", "H310", "H330"))
    if has_carc1 and has_fatal_acute:
        red_flag = True

    if not weights:
        return 0.0, [], red_flag

    score = 1.0 - prod((1.0 - w) for w in weights)
    score = min(max(score, 0.0), 1.0)

    return score, weights, red_flag


@agent.tool_plain
def route_hazard_score(
    compounds: list[CompoundHazard],
    gamma: float = 0.6,
) -> dict[str, Any]:
    """
    Compute route hazard using:

        route_hazard = (1 - gamma) * average(compound_hazard)
                     + gamma * max(compound_hazard)

    Args:
        compounds: list of CompoundHazard objects
        gamma: weight on worst compound, must be between 0 and 1

    Returns:
        dict with route-level and compound-level hazard information
    """

    processed: list[CompoundHazard] = []

    for c in compounds:
        score, matched_weights, red_flag = compound_hazard_score(c.hazard_codes)
        c.matched_weights = matched_weights
        c.red_flag = red_flag
        c.compound_hazard = score
        processed.append(c)

    if not processed:
        return {
            "route_hazard": 0.0,
            "route_safety": 1.0,
            "safety_points": 100.0,
            "average_compound_hazard": 0.0,
            "max_compound_hazard": 0.0,
            "has_red_flag": False,
            "compounds": [],
        }

    scores = [c.compound_hazard for c in processed]
    avg_h = sum(scores) / len(scores)
    max_h = max(scores)

    route_hazard = min(1.0, (1.0 - gamma) * avg_h + gamma * max_h)
    route_safety = 1.0 - route_hazard
    safety_points = 100.0 * route_safety
    has_red_flag = any(c.red_flag for c in processed)

    return {
        "route_hazard": round(route_hazard, 4),
        "route_safety": round(route_safety, 4),
        "safety_points": round(safety_points, 1),
        "average_compound_hazard": round(avg_h, 4),
        "max_compound_hazard": round(max_h, 4),
        "has_red_flag": has_red_flag,
        "gamma": gamma,
        "compounds": [
            {
                "name": c.name,
                "hazard_codes": c.hazard_codes,
                "matched_weights": [round(w, 3) for w in c.matched_weights],
                "compound_hazard": round(c.compound_hazard, 4),
                "red_flag": c.red_flag,
            }
            for c in processed
        ],
    }
