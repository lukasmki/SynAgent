from __future__ import annotations

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

from .validation import agent as validation_agent
from .optimization import agent as optimization_agent
from .chemspace import agent as chemspace_agent
from .corrector import (
    agent as corrector_agent,
    _parse_route_input,
    _strip_tag,
    correct_route as _correct_route,
    design_route as _design_route,
)
from .validation import auto_validate_reaction as _auto_validate_reaction

from ..chemspacetool import ChemspaceDeps, check_building_block_available
from ..tokenmanager import ChemspaceTokenManager

load_dotenv()

MASTER_RPOMPT = """You are the master retrosynthesis workflow agent.

You coordinate four specialist agents:

1. validation agent
   - checks whether reaction SMILES, reaction SMARTS, reactants, products,
     and building blocks are chemically valid.

2. chemspace agent
   - searches ChemSpace for building block price and availability.

3. optimization agent
   - calculates route hazard score from compound hazard codes.

4. corrector agent / auto_resolve_route pipeline
   - auto_resolve_route, auto_correct_and_validate, and auto_design_route are ALL fully
     deterministic Python — they call correct_route/design_route/auto_validate_reaction
     directly, with NO LLM call anywhere inside them, so they cannot hallucinate. Their
     output is ground truth, not a model's paraphrase of a tool result. Trust their
     output completely and report it verbatim — do not "double check" it by separately
     asking the validation agent about the same route, since that would only reintroduce
     the hallucination risk these tools were specifically built to avoid.
   - The validation agent and the plain call_corrector_agent tool are LLM-mediated and
     CAN occasionally misreport or hallucinate. Reserve them only for free-text
     exploration the user explicitly asks for (e.g. "explain the mechanism", "what do
     you think is wrong") — never as the source of truth for whether a route/fix is
     actually valid.
   - IMPORTANT: the corrector agent (reached ONLY via call_corrector_agent, never via
     call_validation_agent) has two real external-lookup tools: literature_lookup
     (searches CrossRef for real papers/DOIs on a reaction class) and pubchem_lookup
     (checks PubChem for a known compound by SMILES, returning its CID, name/synonyms,
     and PubMed/patent cross-references). Whenever the user asks to "search literature",
     "check PubChem", "look this up", or asks why a route can't be found and what to do
     about it after auto_design_route/auto_resolve_route reports no route found, call
     call_corrector_agent (NOT call_validation_agent — it has neither tool) and instruct
     it to actually call literature_lookup and/or pubchem_lookup on the target/intermediate
     SMILES, rather than writing prose that just tells the user to go check those sources
     themselves. Report back whatever those tools actually returned (CIDs, DOIs, etc.) —
     never paraphrase "you should check PubChem" when the agent could have checked it.

Your job is not to do all calculations yourself.
Your job is to decide which specialist agent should be called, call it,
then combine the results into a clear final report.

General workflow:
1. If the user gives a route in <SMILES>,<difficulty>,<JSON> format and wants it
   validated and (if invalid) fixed, call auto_resolve_route with the ENTIRE original
   input, exactly as given — do not extract or paraphrase reactants/products yourself.
   This is THE autonomous entry point: deterministic, no LLM calls inside it, so its
   verdict is ground truth — it validates, and if invalid, automatically redesigns the
   full route, entirely on its own, with no further input from you or the user needed.
   Do NOT manually chain call_validation_agent / call_corrector_agent /
   auto_correct_and_validate / auto_design_route yourself unless the user explicitly
   asks for one specific step in isolation (e.g. "just validate this") — auto_resolve_route
   already runs the full cycle, and re-running pieces through the LLM-mediated tools on
   top of it would only add hallucination risk to an already-trustworthy result.
2. If the user asks for price or availability, call the ChemSpace agent.
3. If the user gives hazard codes or asks for hazard score, call the optimization agent.
4. When reporting auto_resolve_route's result, pass through what it found plainly and
   verbatim:
   - If it was already valid, say so.
   - If it ran the diagnose-and-redesign cycle, report the per-step validation/fix
     findings and the final outcome — a fully validated new route (list every step's
     reactants/product) and any building-block substitutions it flagged, or, if none
     could be found, the honest "no route found" message along with why (e.g. "needs
     chemistry outside the template library"). Never invent a plausible-sounding
     mechanism yourself when the tools report failure, and never claim the validation
     agent confirmed or rejected anything about this result — it wasn't involved.
5. If the user asks for a full route evaluation, call validation, ChemSpace, and optimization,
   then summarize all outputs together.

Do not invent prices.
Do not invent ChemSpace results.
Do not invent hazard codes.
Do not invent reactant fixes — only report what the corrector agent suggested AND what
the validation agent confirmed about that suggestion.
If information is missing, clearly say what is missing.""".strip()

model = GoogleModel("gemini-3.1-flash-lite")
agent = Agent(
    model,
    output_type=str,
    system_prompt=MASTER_RPOMPT,
    deps_type=ChemspaceDeps,

)

def _ensure_model(subagent:Agent) -> Agent:
    if subagent.model is None:
        subagent.model = model
    return subagent

@agent.tool_plain
async def call_validation_agent(user_input: str) -> str:
    """
    call the validation agent to validate reaction pathway information.
    """
    subagent = _ensure_model(validation_agent)
    result = await subagent.run(user_input)
    return str(result.output)


@agent.tool_plain
async def call_chemspace_agent(user_input: str) -> str:
    """
    call the Chemspace agent to search price or availability of compounds
    """
    subagent = _ensure_model(chemspace_agent)

    mgr = ChemspaceTokenManager()
    deps = ChemspaceDeps(mgr=mgr)

    result = await subagent.run(user_input, deps=deps)
    return str(result.output)

@agent.tool_plain
async def call_optimization_agent(user_input:str) -> str:
    """
    Call the optimization agent to calculate route hazard score.
    """
    subagent = _ensure_model(optimization_agent)
    result = await subagent.run(user_input)
    return str(result.output)

@agent.tool_plain
async def call_corrector_agent(user_input: str) -> str:
    """
    Call the corrector agent to PROPOSE a fix for a reaction step that the validation
    agent reported as invalid. The corrector only suggests a modification — its output
    is not a confirmed fix. Always follow this up with revalidate_correction (or a direct
    call_validation_agent call) before treating the suggestion as resolved.

    This is also the ONLY way to reach literature_lookup (real CrossRef paper search) and
    pubchem_lookup (real PubChem compound/CID/xref lookup) — the validation agent does not
    have these tools. If the user wants literature or PubChem checked for a target that
    auto_design_route/auto_resolve_route couldn't find a route for, call this with an
    instruction to actually invoke literature_lookup/pubchem_lookup on the relevant
    SMILES — do not let it fall back to telling the user to search those sources by hand.
    """
    subagent = _ensure_model(corrector_agent)
    result = await subagent.run(user_input)
    return str(result.output)


@agent.tool_plain
async def revalidate_correction(corrector_suggestion: str) -> str:
    """
    Send the corrector agent's modification suggestion to the validation agent so it can
    independently re-validate every corrected step. This is the step that turns a
    corrector "proposal" into a confirmed result — the corrector never confirms its own
    fix.

    Args:
        corrector_suggestion (str): The full text output returned by call_corrector_agent.
    """
    subagent = _ensure_model(validation_agent)
    result = await subagent.run(_revalidation_prompt(corrector_suggestion))
    return str(result.output)


def _revalidation_prompt(corrector_suggestion: str) -> str:
    return (
        "The corrector agent proposed the following modification(s) to a retrosynthesis "
        "route. For every corrected step it proposed, independently re-validate the "
        "corrected reactants against the expected product using auto_validate_reaction "
        "(do not assume the corrector's proposal is correct just because it says so). "
        "Report, per step, whether your own validation confirms the fix is chemically "
        "valid, and if not, say exactly why.\n\n"
        "End your response with exactly one final line, with nothing after it: "
        "'VERDICT: VALID' if every proposed step is confirmed valid, otherwise "
        "'VERDICT: INVALID'.\n\n"
        f"CORRECTOR SUGGESTION:\n{corrector_suggestion}"
    )


def _verdict_is_valid(revalidation_text: str) -> bool:
    for line in reversed(revalidation_text.strip().splitlines()):
        line = line.strip().upper()
        if line.startswith("VERDICT:"):
            return "INVALID" not in line and "VALID" in line
    return False


def _format_correct_route_result(result: dict) -> str:
    """Builds the report directly from correct_route's return dict — no LLM involved,
    so this can never contradict what the deterministic per-step check actually found."""
    if result.get("error"):
        return f"error: {result['error']}"

    lines = [
        f"all_resolved: {result['all_resolved']}, "
        f"chain_consistent: {result.get('chain_consistent')}"
    ]
    for step in result["steps"]:
        line = (
            f"  Step {step['reaction_number']} [{step['status']}]: "
            f"{step['original_reactants']} -> {step['product']}"
        )
        if step["status"] == "fixed":
            line += f" | corrected_reactants: {step['corrected_reactants']} (matched_smarts: {step['matched_smarts']})"
        if step.get("chain_warning"):
            line += f" | CHAIN WARNING: {step['chain_warning']}"
        line += f" | {step['message']}"
        lines.append(line)
    return "\n".join(lines)


@agent.tool_plain
def auto_correct_and_validate(route_input: str) -> str:
    """
    Deterministically validates a route and attempts a single-step fix for any invalid
    step — entirely in Python, no LLM calls anywhere, so there is zero hallucination
    risk. Calls correct_route directly; every "fixed" status it returns is already
    guaranteed valid because correct_route internally re-confirms each fix via
    auto_validate_reaction before returning it.

    There is no retry loop: correct_route's search already exhaustively tries every
    common SMARTS template and reactant permutation in a single call, so retrying
    couldn't find anything a first attempt missed. If a step comes back "unfixable", or
    chain_consistent is False, escalate to auto_design_route for a full redesign instead
    of retrying this tool.

    Args:
        route_input (str): The full route input, exactly as the user gave it, i.e.
            "<SMILES>,<difficulty>,<JSON>".
    """
    result = _correct_route(route_input)
    return _format_correct_route_result(result)


def _format_design_route_result(result: dict) -> str:
    """Builds the report directly from design_route's return dict — no LLM transcription
    step, so this text can never contradict what the deterministic search actually
    found (route_found, steps, swaps are all ground truth, not a paraphrase)."""
    if not result["route_found"]:
        return f"route_found: False\n{result['message']}"

    lines = [f"route_found: True ({len(result['steps'])} step(s))"]
    for step in result["steps"]:
        lines.append(
            f"  Step {step['reaction_number']}: {step['reactants']} -> {step['product']} "
            f"(matched_smarts: {step['matched_smarts']})"
        )
    if result.get("recommended_building_blocks"):
        lines.append(
            "  NOTE: the originally given building blocks were not enough on their own. "
            "This route requires additionally sourcing these exact molecule(s) instead:"
        )
        for block in result["recommended_building_blocks"]:
            lines.append(f"    - {block!r}")
    if result["building_block_swaps"]:
        lines.append(
            "  WARNING: even after that, this route is still NOT fully built from exact "
            "matches. The following remaining substitution(s) are assumed:"
        )
        for swap in result["building_block_swaps"]:
            lines.append(
                f"    - needed {swap['needed_smiles']!r}, substituting "
                f"{swap['suggested_building_block']!r} (similarity {swap['similarity']})"
            )
    lines.append(result["message"])
    return "\n".join(lines)


async def _check_leaf_availability(steps: list[dict]) -> str:
    """Cross-checks every true starting-material leaf in a route (reactants not
    produced by any other step) against ChemSpace for real purchasability — a
    structurally "terminal" molecule in the search isn't necessarily something you can
    actually buy. Never raises: a network/API failure is reported per-block as
    unconfirmed, never silently treated as available."""
    produced = {step["product"] for step in steps}
    leaves = sorted({r for step in steps for r in step["reactants"] if r not in produced})
    if not leaves:
        return ""

    mgr = ChemspaceTokenManager()
    deps = ChemspaceDeps(mgr=mgr)
    lines = ["", "BUILDING BLOCK AVAILABILITY (ChemSpace exact-match check):"]
    for smi in leaves:
        result = await check_building_block_available(deps, smi)
        if result["error"]:
            lines.append(f"  - {smi!r}: UNCONFIRMED (lookup failed: {result['error']})")
        elif result["available"]:
            lines.append(f"  - {smi!r}: available ({result['vendor_count']} vendor match(es))")
        else:
            lines.append(f"  - {smi!r}: NOT FOUND on ChemSpace — may not be a real purchasable building block")
    return "\n".join(lines)


@agent.tool_plain
async def auto_design_route(target_smiles: str, known_smiles_csv: str, max_depth: int = 4) -> str:
    """
    Autonomously designs a COMPLETE multi-step route from target_smiles down to the
    given building blocks — use this when a single-step fix (auto_correct_and_validate)
    leaves the route disconnected, or when the user explicitly asks for a full/best route
    to be found or redesigned from scratch, rather than asking the corrector to reason
    about mechanisms in prose.

    Entirely deterministic — no LLM is involved anywhere in this call, so there is no
    hallucination risk. Calls design_route directly in Python, then independently
    re-confirms every returned step with a second, separate call to
    auto_validate_reaction (belt-and-suspenders: design_route already validated each
    step once during its search; this re-checks the exact final result one more time).
    If the originally given building blocks aren't quite right (e.g. a substituent in
    the wrong place), design_route automatically retries with the EXACT molecule the
    search determined it actually needs, converging on a fully resolved route where
    possible — that final substitution is reported in "recommended_building_blocks", so
    you always know exactly what to source instead of what was given. Only if the search
    still can't close the gap after that does it fall back to a flagged similarity-based
    "building_block_swaps" guess as a last resort.

    Once a route is confirmed, every true starting-material leaf (not just structurally
    "terminal" in the search, but the actual reactants nothing else produces) is also
    cross-checked against ChemSpace for real purchasability — a route can be fully
    chemically valid yet still need a starting material nobody sells.

    Args:
        target_smiles (str): The target molecule to design a route for.
        known_smiles_csv (str): Comma-separated SMILES of the available building blocks.
        max_depth (int): Maximum number of reaction steps to search before giving up.
    """
    known_smiles = [s.strip() for s in known_smiles_csv.split(",") if s.strip()]
    result = _design_route(target_smiles, known_smiles, max_depth=max_depth)
    design_text = _format_design_route_result(result)

    if not result["route_found"]:
        return f"DESIGN_ROUTE RESULT (route_found: False):\n\n{design_text}"

    recheck_failures = []
    for step in result["steps"]:
        ok, msg = _auto_validate_reaction(step["reactants"], step["product"])
        if not ok:
            recheck_failures.append(f"Step {step['reaction_number']}: {msg}")

    if recheck_failures:
        return (
            f"DESIGN_ROUTE RESULT (RE-CHECK FAILED — do not treat as resolved):\n\n"
            f"{design_text}\n\n"
            f"Independent re-check found problems: {recheck_failures}"
        )

    availability_text = await _check_leaf_availability(result["steps"])

    return (
        f"DESIGN_ROUTE RESULT (CONFIRMED — every step independently re-checked via "
        f"auto_validate_reaction, deterministically, no LLM involved):\n\n{design_text}"
        f"{availability_text}"
    )


def _extract_target_and_blocks(route_input: str) -> tuple[str | None, list[str]]:
    """Deterministically pulls the target SMILES and building_blocks list out of a
    <SMILES>,<difficulty>,<JSON> route input, the same way the corrector parses routes —
    so auto_resolve_route never has to ask an LLM to transcribe these by hand."""
    target_smiles = route_input.strip().split(",", 1)[0].strip() or None
    try:
        data = _parse_route_input(route_input)
    except Exception:
        return target_smiles, []
    if not isinstance(data, dict):
        return target_smiles, []
    blocks = [
        _strip_tag(b, "<bb>", "</bb>")
        for b in data.get("building_blocks", [])
        if isinstance(b, str) and b.strip()
    ]
    return target_smiles, blocks


@agent.tool_plain
async def auto_resolve_route(route_input: str, max_depth: int = 4) -> str:
    """
    Fully autonomous entry point for a route the user wants validated and, if invalid,
    fixed — runs the entire validate -> diagnose -> redesign cycle on its own with no
    further user input needed. Entirely deterministic: no LLM is called anywhere in this
    function, so there is zero hallucination risk — every claim in the output traces
    directly to a tool's return value.

    Call call_validation_agent as a SEPARATE prior tool call if you also want the LLM's
    opinion shown to the user as its own distinct step — auto_resolve_route itself no
    longer calls it internally, so the two show up as two separate tool invocations.

    Steps:
    1. Calls correct_route (deterministic, Python, no LLM) to validate the route as
       given and attempt a single-step fix for any invalid step. THIS is the
       authoritative validity check.
    2. If every step is already valid AND chain_consistent, returns that immediately.
    3. Otherwise, the per-step "message"/"chain_warning" fields from correct_route ARE
       the diagnosis — no separate LLM diagnosis call is made or needed.
    4. Calls auto_design_route (also deterministic) on the target molecule (parsed from
       route_input) against the route's own building_blocks, to find a complete,
       independently re-checked route from scratch.
    5. Returns one combined report. Never invents a route itself — if no route is found,
       says so plainly using the tools' own messages.

    Args:
        route_input (str): The full route input exactly as the user gave it, i.e.
            "<SMILES>,<difficulty>,<JSON>".
        max_depth (int): Maximum number of reaction steps auto_design_route may search.
    """
    correction = _correct_route(route_input)
    correction_text = _format_correct_route_result(correction)

    if correction.get("error"):
        return f"ROUTE COULD NOT BE PARSED.\n\n{correction_text}"

    if correction["all_resolved"] and correction.get("chain_consistent"):
        return f"ROUTE ALREADY VALID.\n\n{correction_text}"

    target_smiles, building_blocks = _extract_target_and_blocks(route_input)
    if target_smiles is None:
        return (
            f"ROUTE INVALID and could not be redesigned: no target molecule SMILES "
            f"could be parsed from the input.\n\n{correction_text}"
        )

    design_report = await auto_design_route(
        target_smiles=target_smiles,
        known_smiles_csv=", ".join(building_blocks),
        max_depth=max_depth,
    )

    return (
        f"ROUTE INVALID — ran autonomous diagnose-and-redesign cycle.\n\n"
        f"PER-STEP VALIDATION/FIX ATTEMPT (correct_route):\n{correction_text}\n\n"
        f"{design_report}"
    )


@agent.tool_plain
async def full_route_evaluation(
    pathway_input: str,
    chemspace_query: str,
    hazard_input: str,
) -> str:
    validation_result = await call_validation_agent(pathway_input)
    chemspace_result = await call_chemspace_agent(chemspace_query)
    optimization_result = await call_optimization_agent(hazard_input)

    return f"""
FULL ROUTE EVALUATION

1. VALIDATION RESULT
{validation_result}

2. CHEMSPACE PRICE / AVAILABILITY RESULT
{chemspace_result}

3. HAZARD OPTIMIZATION RESULT
{optimization_result}
""".strip()


    
