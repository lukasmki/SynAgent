#!/usr/bin/env python
"""Ablation: how much does each corrector tool actually contribute?

Measured directly against the full frozen ChEMBL subset -- no LLM agent loop
needed, because the underlying fix-finding logic (fix_smiles,
fix_template, fix_via_analogue_building_block, retro_disconnection,
retro_disconnection_all_templates, partial_reactant_retention) is pure RDKit
search with no model call in it. That means this doesn't depend on whether a
particular orchestrator (Qwen, Gemma, ...) happens to discover and call each
tool -- it measures each tool's actual fix-finding capability directly, which
the cluster run's tool-usage numbers could not do (fix_step calls these
internally as plain Python, invisible to any tool-call tracker built from
conversation history -- see table1_with_synagent.md for that whole story).

Cumulative waterfall, one tool added at a time, in the exact order fix_step's
auto chain tries them (corrector/_toolset.py): strict-preserving tools
first (never invent outside RXN1, never touch the declared product -- a fix
found here is always an exact match), analog-only tools last (fallback once
every strict option has failed):

  1_fix_smiles                     SMILES-level hygiene: fix_building_blocks'
                                    building-block repair + fix_smiles for
                                    invalid_reactant_smiles/invalid_product_smiles
                                    steps. No template-level repair at all yet.
  2_fix_template                  + RXN1 template search against the ORIGINAL
                                    reactants (91 templates)
  3_retro_disconnection          + reverse the step's OWN template against the
                                    product, retry RXN1 on the fragments
  4_retro_disconnection_all_templates
                                  + reverse ALL 91 RXN1 templates against the
                                    product (recovers from the step having had
                                    the wrong template to begin with, not just
                                    the wrong reactants)
  5_partial_reactant_retention   + keep whichever original reactant is
                                    plausibly correct, retro-derive a
                                    replacement only for the other one, across
                                    all 91 templates -- the middle ground
                                    between fix_template (keeps everything)
                                    and retro_disconnection_all_templates
                                    (discards everything). Still strict.
  6_rescue_unparseable_via_retro + for a response that does not parse at all
                                    (no route for any other tool to work on),
                                    build a one-step route to the target from
                                    scratch by searching all 91 RXN1 templates
                                    in reverse. The 1b-2m output has 10 such
                                    rows, recorded as the literal string
                                    "json format error" -- an upstream
                                    generation failure, not malformed JSON, so
                                    there is no prefix to salvage. Strict: the
                                    template is in RXN1, the reaction is
                                    RDKit-verified, and the declared product
                                    is the original target. The only tier that
                                    can move Valid JSON.

The two tiers below are OPT-IN (--include-analog-tools), not part of the
default waterfall: fix_step reaches them only on an explicit method= call, so
the SynLlama comparison, whose every number must come from exact matches,
does not include them.

  6_fix_via_analogue_building_block
                                  + swap one reactant for a database analogue,
                                    retry RXN1. The substituted reactant makes
                                    the reaction's product differ from the
                                    declared one, so it lifts Matched
                                    Reactants without earning a Good Product.
                                    Tried only once every strict-preserving
                                    tool above has failed.
  7_fix_via_product_analogue_retro
                                  + relax the TARGET PRODUCT itself: find
                                    purchasable molecules similar to it, then
                                    run the full 91-template retro search
                                    aiming at each candidate instead of the
                                    original. Every prior tool keeps the
                                    original target fixed and only varies the
                                    ingredients feeding into it; this is the
                                    only one that asks "is there an easier,
                                    similar target instead" -- recovers routes
                                    where the original product has no RXN1
                                    path at all. Same scoring consequence as
                                    fix_via_analogue_building_block: the
                                    declared product stays the original
                                    target, so it never counts as strict.

fix_smarts (syntax repair for invalid_template steps) was removed entirely:
measured against the real SynLlama 1b-2m checkpoint output, zero of the 1,672
reactions ever have failure_mode invalid_template -- SynLlama reproduces the
91 template STRINGS correctly essentially always (that's Template Mem.
99.82%), so this tool never had anything to fix on that data. invalid_template
steps now fall straight to fix_template.

Putting the two analog-only tools (6, 7) last instead of interleaved with the
strict tools is a deliberate fix, not the original chain order: analog tools
can "claim" a step that a later strict tool would have fixed exactly,
silently converting an available exact match into an analog-only one and
diluting Good Products for no benefit.

extract_template_from_reaction is NOT part of the waterfall above: fix_step
never calls it (by design -- it invents a SMARTS outside RXN1, which fixes
more routes but can never count as "memorized", see table1_with_synagent.md
footnote 4/6) and it's documented to occasionally hang past an hour on a
single pathological input (corrector/_toolset.py). Pass
--include-extract-template to add it as an optional final
8_extract_template_from_reaction tier anyway, each call wrapped in its own
timeout so one bad molecule can't stall the whole run -- off by default so a
plain run stays fast and bounded.

Every reconstruction step below mirrors apply_fixes (corrector/_toolset.py)
exactly, function for function, so "what would apply_fixes have produced
under this tool set" is a faithful question, not an approximation.
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
csv.field_size_limit(10**7)

sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, RDLogger, rdBase  # noqa: E402

RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.*")

from synagent.corrector._toolset import (  # noqa: E402
    CorrectorToolset,
    _fix_template_sync,
    _fix_via_analogue_sync,
    _retro_disconnection_sync,
    _retro_disconnection_all_templates_sync,
    _partial_reactant_retention_sync,
    _fix_via_product_analogue_retro_sync,
    _rxn1_templates,
    _try_parse_smarts,
)
from synagent.validation._toolset import (  # noqa: E402
    _parse_route_json,
    _validate_route_dict,
)
from itertools import permutations  # noqa: E402

# Strict-preserving tools (never invent outside RXN1, never touch the
# declared product) are ordered FIRST; the two analog-only tools
# (fix_via_analogue_building_block, fix_via_product_analogue_retro) are last,
# as a fallback only. Getting this backwards has a real, measured cost: with
# fix_via_analogue_building_block before retro_disconnection_all_templates
# and partial_reactant_retention, it could "claim" a step those two would
# have fixed exactly, silently converting an available exact match into an
# analog-only one -- see corrector/_toolset.py's fix_step docstring.
TOOL_ORDER = [
    "fix_smiles",
    "fix_template",
    "retro_disconnection",
    "retro_disconnection_all_templates",
    "partial_reactant_retention",
    # Whole-route rather than per-reaction, and deliberately LAST so tiers
    # 1-5 keep reporting exactly the marginal contributions they did before
    # it existed. It is the only tier that can move Valid JSON: every other
    # tool needs a parsed route to work on, and these rows have none.
    "rescue_unparseable_via_retro",
]

# The two analog-only tools are NOT part of the default waterfall. They are
# explicit-method-only in fix_step -- never a silent default -- so they are
# not in the SynLlama comparison either: every number that comparison reports
# has to come from a chain that only ever produces exact matches. Pass
# --include-analog-tools to append them as two further tiers.
ANALOG_TOOL_ORDER = [
    "fix_via_analogue_building_block",
    "fix_via_product_analogue_retro",
]

EXTRACT_TEMPLATE_TIMEOUT_S = 30


def _cumulative_variants(extra_tools: list[str] | None = None) -> dict[str, set[str]]:
    order = TOOL_ORDER + (extra_tools or [])
    variants: dict[str, set[str]] = {}
    enabled: set[str] = set()
    for i, tool in enumerate(order, start=1):
        enabled = enabled | {tool}
        variants[f"{i}_{tool}"] = set(enabled)
    return variants


def strip(value: str, tag: str) -> str:
    start, end = f"<{tag}>", f"</{tag}>"
    if value.startswith(start):
        value = value[len(start):]
    if value.endswith(end):
        value = value[: -len(end)]
    return value


def _canon(smiles: str) -> str:
    """Canonical SMILES for BB Selection's set comparison, falling back to the
    raw string on a parse failure so an invalid SMILES still participates in
    the comparison instead of silently vanishing."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True) if mol is not None else smiles
    except Exception:
        return smiles


def _largest_reactant_ratio(product: str, reactants: list[str]) -> float | None:
    """Largest proposed reactant as a fraction of the product's heavy-atom
    count. A genuine retrosynthetic disconnection roughly halves the molecule,
    so this lands near 0.5-0.6; a value near 1.0 means the 'reactant' is the
    product with a small edit, which is a functional-group interconversion
    dressed up as a disconnection."""
    pm = Chem.MolFromSmiles(product)
    if pm is None:
        return None
    n_p = pm.GetNumHeavyAtoms()
    if n_p == 0:
        return None
    best = 0
    for r in reactants or []:
        m = Chem.MolFromSmiles(r)
        if m is not None:
            best = max(best, m.GetNumHeavyAtoms())
    return best / n_p if best else None


def _is_degenerate(product: str, reactants: list[str], ratio_cap: float | None = None,
                   min_reactants: int | None = None) -> bool:
    """Reject a proposal that does not actually simplify the target.

    Measured on the official 1b-2m output: retro_disconnection_all_templates'
    repairs have a median largest-reactant/product ratio of 0.93, 59% leave a
    reactant >=90% of the product's size and 44% are single-reactant, against
    0.70 / 22% / 20% for SynLlama's OWN passing reactions. The tool searches
    all 91 templates for anything that reaches the product and takes the first
    hit, and the easiest hits are the ones that barely change the molecule.
    Those repairs lift Matched Reactants and Good Products almost for free --
    a reactant 93% the size of the product will of course react to give it --
    while making the route worse than what the model produced unaided.

    Two criteria, both off by default:

    min_reactants -- the recommended one. A "disconnection" that consumes no
    reagent is a functional-group interconversion, not a synthesis step, and
    single-reactant matches are the corrector's actual bad habit: 44% of its
    repairs against 20% of the baseline's own reactions.

    ratio_cap -- blunter, and it mis-fires. An acylation (alcohol + CC(=O)O ->
    acetate) scores 0.95+ only because acetic acid is small, yet it is ordinary
    late-stage chemistry that SynLlama's own routes contain at 22%. Two of the
    ten rescuable placeholder targets are exactly that, and a 0.9 cap discards
    both. Kept for comparison; not recommended on its own.
    """
    if min_reactants is not None and len([r for r in (reactants or []) if r]) < min_reactants:
        return True
    if ratio_cap is None:
        return False
    ratio = _largest_reactant_ratio(product, reactants)
    # >=, not >, so the cut matches the audit figure it was chosen from: 59%
    # of tier-4 repairs leave a reactant >=90% of the product's heavy atoms.
    return ratio is not None and ratio >= ratio_cap


async def fix_one_reaction(
    ts: CorrectorToolset, rxn, enabled: set[str], ratio_cap: float | None = None,
    min_reactants: int | None = None,
) -> dict | None:
    """Try to fix one failed reaction using only the tools in `enabled` --
    same order, same functions as fix_step's auto chain, just gated per tool
    so each can be switched on independently for the waterfall.

    `ratio_cap` rejects proposals that do not simplify the target (see
    _is_degenerate). A rejected proposal falls through to the next tool rather
    than ending the chain, so a later tool can still offer a real one."""
    failure = rxn.failure_mode
    template = rxn.reaction_template
    reactants = rxn.reactant_smiles
    product = rxn.expected_product

    if failure == "invalid_template":
        if "fix_template" in enabled:
            tr = _fix_template_sync(reactants, product)
            if tr.get("found"):
                return {"new_template": tr["template"]}
        return None

    if failure in ("no_products", "wrong_product"):
        # Strict tools first (see the ordering note above TOOL_ORDER), analog
        # tools last.
        if "fix_template" in enabled:
            tr = _fix_template_sync(reactants, product)
            if tr.get("found"):
                return {"new_template": tr["template"]}
        if "retro_disconnection" in enabled:
            rr = _retro_disconnection_sync(reactants, product, template)
            if rr.get("found") and not _is_degenerate(product, rr["new_reactants"], ratio_cap, min_reactants):
                return {"new_template": rr["template"], "new_reactants": rr["new_reactants"]}
        if "retro_disconnection_all_templates" in enabled:
            # The cap goes INTO the search: a near-copy match no longer ends
            # it, so a real disconnection deeper in the same fragment-set list
            # can still be found. The outer _is_degenerate stays as a backstop
            # for the last-resort fallback the search returns when nothing
            # clean exists.
            rrt = _retro_disconnection_all_templates_sync(
                product, max_reactant_ratio=ratio_cap, min_reactants=min_reactants)
            if rrt.get("found") and not _is_degenerate(product, rrt["new_reactants"], ratio_cap, min_reactants):
                return {"new_template": rrt["template"], "new_reactants": rrt["new_reactants"]}
        if "partial_reactant_retention" in enabled:
            prr = _partial_reactant_retention_sync(reactants, product)
            if prr.get("found") and not _is_degenerate(product, prr["new_reactants"], ratio_cap, min_reactants):
                return {"new_template": prr["template"], "new_reactants": prr["new_reactants"]}
        if "fix_via_analogue_building_block" in enabled:
            ar = _fix_via_analogue_sync(reactants, product)
            if ar.get("found"):
                return {"new_template": ar["template"], "new_reactants": ar["new_reactants"]}
        if "fix_via_product_analogue_retro" in enabled:
            # Deliberately no new_product here: the declared product in
            # build_corrected_route must stay the ORIGINAL target so
            # score_effective_dataset's strict check (actual output vs.
            # declared product) correctly falls through to analog-only,
            # same convention as fix_via_analogue_building_block above.
            par = _fix_via_product_analogue_retro_sync(product, reactants, template)
            if par.get("found"):
                return {"new_template": par["template"], "new_reactants": par["new_reactants"]}
        if "extract_template_from_reaction" in enabled:
            try:
                er = await asyncio.wait_for(
                    ts.extract_template_from_reaction(reactants, product),
                    timeout=EXTRACT_TEMPLATE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                er = {"fixed": False}
            if er.get("fixed") and er.get("self_consistent"):
                return {"new_template": er["smarts"]}
        return None

    if failure in ("invalid_reactant_smiles", "invalid_product_smiles"):
        if "fix_smiles" not in enabled:
            return None
        smiles_to_fix = reactants if failure == "invalid_reactant_smiles" else [product]
        sr = await ts.fix_smiles(smiles_to_fix)
        fixed = {k: v["canonical"] for k, v in sr.items() if v.get("valid") and v.get("canonical")}
        return {"smiles_fixes": fixed} if fixed else None

    return None


async def build_corrected_route(
    ts: CorrectorToolset, original_response: str, enabled: set[str],
    target_smiles: str | None = None, ratio_cap: float | None = None,
    min_reactants: int | None = None,
) -> dict | None:
    """apply_fixes-equivalent reconstruction for one route under one tool-set
    variant. Returns None if the route already passes (nothing to correct),
    or if it can't be parsed and rescue_unparseable_via_retro is off."""
    try:
        route = _parse_route_json(original_response)
    except Exception:
        # No route to repair -- every other tool in the chain works on a
        # parsed route and has nothing to bite on. The 1b-2m output records
        # 10 of these as the literal string "json format error", an upstream
        # generation failure rather than malformed JSON, so there is not even
        # a truncated prefix to salvage (cf. fix_invalid_json.py, which
        # handles the truncation case by keeping the complete prefix).
        #
        # What IS available is the target itself. retro_disconnection_all_
        # templates searches all 91 RXN1 templates in reverse from a bare
        # product, so it can build a one-step route with nothing else in
        # hand. Strictly in-set, RDKit-verified, and the declared product is
        # the original target -- so a route built here is scored exactly like
        # any other, with no analog allowance.
        if "rescue_unparseable_via_retro" not in enabled or not target_smiles:
            return None
        rr = _retro_disconnection_all_templates_sync(
            target_smiles, max_reactant_ratio=ratio_cap, min_reactants=min_reactants)
        if not rr.get("found") or _is_degenerate(target_smiles, rr["new_reactants"], ratio_cap, min_reactants):
            return None
        return {
            "reactions": [{
                "reaction_number": 1,
                "reaction_template": rr["template"],
                "reactants": rr["new_reactants"],
                "product": target_smiles,
            }],
            "building_blocks": sorted(set(rr["new_reactants"])),
        }
    try:
        report = _validate_route_dict(route, analog_product_threshold=None)
    except Exception:
        return None
    if report.all_building_blocks_valid and report.all_reactions_passed:
        return None

    bb_fixes: dict[str, str] = {}
    if "fix_smiles" in enabled:
        for bb in report.building_blocks:
            if not bb.is_valid:
                sr = await ts.fix_smiles([bb.smiles])
                r = sr.get(bb.smiles, {})
                if r.get("valid") and r.get("canonical"):
                    bb_fixes[bb.smiles] = r["canonical"]

    corrected_reactions = []
    for rxn in report.reactions:
        if rxn.status == "passed":
            corrected_reactions.append({
                "reaction_number": rxn.reaction_number,
                "reaction_template": rxn.reaction_template,
                "reactants": rxn.reactant_smiles,
                "product": rxn.expected_product,
            })
            continue

        fix = await fix_one_reaction(ts, rxn, enabled, ratio_cap=ratio_cap,
                                     min_reactants=min_reactants) or {}
        template = fix.get("new_template") or rxn.reaction_template
        if fix.get("new_reactants"):
            reactants = fix["new_reactants"]
        else:
            smiles_fixes = fix.get("smiles_fixes", {})
            reactants = [
                smiles_fixes.get(r) or bb_fixes.get(r) or r for r in rxn.reactant_smiles
            ]
        smiles_fixes = fix.get("smiles_fixes", {})
        product = (
            smiles_fixes.get(rxn.expected_product)
            or bb_fixes.get(rxn.expected_product)
            or rxn.expected_product
        )
        corrected_reactions.append({
            "reaction_number": rxn.reaction_number,
            "reaction_template": template,
            "reactants": reactants,
            "product": product,
        })

    # Same derivation apply_fixes uses: a building block is a reactant that
    # isn't another step's product.
    seen: list[str] = []
    seen_set: set[str] = set()
    products: set[str] = set()
    for rxn in corrected_reactions:
        for r in rxn["reactants"]:
            if r and r not in seen_set:
                seen_set.add(r)
                seen.append(r)
        if rxn["product"]:
            products.add(rxn["product"])
    corrected_bbs = [r for r in seen if r not in products]

    return {"reactions": corrected_reactions, "building_blocks": corrected_bbs}


_RXN1_TEMPLATE_SET = set(_rxn1_templates())


def _arrange_and_react(template: str, reactant_smis: list[str]):
    """Faithful reimplementation of SynLlama's own
    arrange_reactants_and_react_synllama (steps/step_30_0_benchmark_filter_raw_output.py):
    only accepts an exact reactant-count match against the template's arity,
    tries every permutation, and returns the raw RDKit product mols (None if
    the count doesn't match or nothing reacts)."""
    rxn = _try_parse_smarts(template)
    if rxn is None:
        return None
    mols = [Chem.MolFromSmiles(s) for s in reactant_smis]
    if any(m is None for m in mols):
        return None
    if len(mols) != rxn.GetNumReactantTemplates():
        return None
    prods = []
    for perm in permutations(mols):
        try:
            for outs in rxn.RunReactants(perm):
                for m in outs:
                    try:
                        Chem.SanitizeMol(m)
                        prods.append(m)
                    except Exception:
                        continue
        except Exception:
            continue
    return prods if prods else None


def score_effective_dataset(targets_and_responses: list[tuple[str, str]]) -> dict:
    """Six Table 1 metrics computed with SynLlama's OWN exact scoring
    algorithm (calc_benchmark_rxn in their steps/step_30_0_benchmark_filter_raw_output.py,
    github.com/THGLab/SynLlama), reimplemented faithfully rather than
    approximated. Verified to reproduce every one of the paper's own reported
    1b-2m numbers exactly (valid_responses 99.0, valid_smiles 95.23,
    recycled_bbs 99.47, template_memorization 99.82, matched_reactants 70.93,
    good_products 87.02, plus the two hidden llm_benchmark_stats.csv columns
    total_success_formats 97.9 and total_success_reactions 56.9), on
    data/synllama-official-1b2m.csv scored with json.loads only (no repair).

    Key differences from an intuitive reimplementation, all confirmed by
    reading their source directly rather than guessing: (1) Valid SMILES only
    ever checks reactants and products INSIDE `reactions` -- building_blocks
    are never separately validity-checked. (2) BB Selection ("recycled_bbs")
    is a FRACTIONAL per-target average of "is this declared building block
    actually used somewhere in the route" (extras are fine, but an unused
    declared building block costs partial credit) -- not a route-level
    binary pass/fail, and not "every used reactant must be declared" (the
    opposite direction). (3) Matched Reactants requires an EXACT reactant
    COUNT match against the template's arity (no permutation search across a
    different reactant count) before even trying the reaction. (4) Good
    Products divides by successful (matched) reactions, not by all
    reactions -- the paper's own number is already conditional.

    Good Products is exact-match ONLY, as in their algorithm: canonical-SMILES
    membership of the declared product in the reaction's outputs. An earlier
    version of this function also reported a good_products_analog_percent
    (same denominator, Morgan/Tanimoto similarity when the exact check fails).
    It has been removed: it has no counterpart in SynLlama's algorithm, so it
    was never comparable to anything in the paper, and reporting it beside the
    exact figure invited the two to be read as one metric.
    """
    n_targets = len(targets_and_responses)
    successful_trials = 0
    total_reactions = 0
    successful_reactions = 0
    not_in_template = 0
    invalid_smiles_count = 0
    total_molecules = 0
    n_products_strict = 0
    bb_obedience_values: list[float] = []

    for target_smiles, resp in targets_and_responses:
        try:
            output = json.loads(resp)
        except Exception:
            continue
        if not isinstance(output, dict) or "reactions" not in output or "building_blocks" not in output:
            continue

        reactions = output["reactions"]
        building_blocks = output["building_blocks"]
        reactant_stack = [target_smiles]
        successful_trials += 1

        aborted = False
        for reaction in reactions:
            if "reaction_template" not in reaction or "reactants" not in reaction or "product" not in reaction:
                successful_trials -= 1
                aborted = True
                break
            raw_template = str(reaction["reaction_template"])
            has_open = "<rxn>" in raw_template
            has_close = "</rxn>" in raw_template
            if has_open and has_close:
                template = raw_template.split("<rxn>")[1].split("</rxn>")[0]
            elif not has_open and not has_close:
                # Corrected routes (build_corrected_route) store templates as
                # bare SMARTS -- our own internal representation, never
                # literal model text, so an untagged template isn't a real
                # format failure the way it would be on raw LLM output (where
                # SynLlama always emits the <rxn> wrapper; see
                # calc_benchmark_rxn's template_no_rxn_tag counter).
                #
                # This branch deliberately does NOT require ">>". An untagged
                # template that is not a well-formed reaction SMARTS has to
                # score the same as the <rxn>-wrapped form of that identical
                # string, which is accepted just above and then simply fails
                # the RXN1 membership test below -- counting against Template
                # Memorization, not against Valid JSON. One route in the
                # official 1b-2m output really does carry a wrapped template
                # with no ">>" in it at all (target
                # Oc1c(Cl)cc(-c2nc(-c3ccccc3)c(-c3cccs3)[nH]2)cc1Cl, reaction
                # 2), so requiring ">>" here penalised the corrector for
                # unwrapping a defect the raw route was forgiven: a phantom
                # -0.1pp on Valid JSON that no tool had caused.
                template = raw_template
            else:
                # Exactly one of the two tags: genuinely malformed model text,
                # and never something build_corrected_route can emit.
                successful_trials -= 1
                aborted = True
                break
            total_reactions += 1
            if template not in _RXN1_TEMPLATE_SET:
                not_in_template += 1
                continue

            reactants = [
                str(r).split("<bb>")[-1].split("</bb>")[0] if "<bb>" in str(r) else str(r)
                for r in reaction["reactants"]
            ]
            reactant_stack.extend(reactants)
            product = str(reaction["product"])
            if product in reactant_stack:
                reactant_stack.remove(product)

            reactant_valid_smis = []
            total_molecules += len(reactants)
            for r in reactants:
                if not (Chem.MolFromSmiles(r) is not None):
                    invalid_smiles_count += 1
                elif r == "":
                    total_molecules -= 1
                    continue
                else:
                    reactant_valid_smis.append(r)
            total_molecules += 1
            product_mol = Chem.MolFromSmiles(product)
            if product_mol is None:
                invalid_smiles_count += 1

            prods = _arrange_and_react(template, reactant_valid_smis)
            if prods is None:
                continue
            successful_reactions += 1

            prod_canon_set = set()
            for p in prods:
                try:
                    prod_canon_set.add(Chem.MolToSmiles(p))
                except Exception:
                    continue
            product_canon = Chem.MolToSmiles(product_mol) if product_mol is not None else None
            if product_mol is not None and product_canon in prod_canon_set:
                n_products_strict += 1

        if aborted:
            continue

        bb_count = 0
        for bb in building_blocks:
            bb_clean = str(bb).split("<bb>")[-1].split("</bb>")[0]
            if bb_clean in reactant_stack:
                bb_count += 1
        bb_obedience_values.append(bb_count / len(building_blocks) if building_blocks else 1.0)

    def pct(n, d):
        return round(n / d * 100, 2) if d else None

    return {
        "valid_json_percent": pct(successful_trials, n_targets),
        "template_mem_percent": pct(total_reactions - not_in_template, total_reactions),
        "bb_selection_percent": (
            round(sum(bb_obedience_values) / len(bb_obedience_values) * 100, 2)
            if bb_obedience_values else None
        ),
        "valid_smiles_percent": pct(total_molecules - invalid_smiles_count, total_molecules),
        "matched_reactants_percent": pct(successful_reactions, total_reactions),
        "good_products_strict_percent": pct(n_products_strict, successful_reactions),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=1400,
                    help="skip failing routes longer than this -- matches "
                         "run_repair_frozen.py's default, for direct comparability. "
                         "Note this gates only which routes the CORRECTOR attempts; "
                         "0_before is always scored over every response, so raising it "
                         "cannot move the baseline. At the default, 93 routes (lengths "
                         "1406-2822) are never corrected, and they hold 149 of the 226 "
                         "invalid SMILES in the 1b-2m output -- 130 of them repairable, "
                         "worth +2.74pp of Valid SMILES.")
    ap.add_argument("--min-reactants", type=int, default=None,
                    help="reject a proposed disconnection with fewer than this many "
                         "reactants (use 2). The recommended screen: a repair that "
                         "consumes no reagent is a functional-group interconversion, "
                         "not a synthesis step. Single-reactant matches are 44%% of the "
                         "corrector's repairs against 20%% of the baseline's own "
                         "reactions. Preferred over --max-reactant-ratio, which also "
                         "discards legitimate acylations (alcohol + CC(=O)O -> acetate "
                         "scores 0.95+ purely because acetic acid is small).")
    ap.add_argument("--max-reactant-ratio", type=float, default=None,
                    help="reject a proposed disconnection whose largest reactant reaches "
                         "this fraction of the product's heavy-atom count (try 0.9). Off "
                         "by default. Guards against repairs that raise Matched Reactants "
                         "and Good Products without simplifying the target -- see "
                         "_is_degenerate for the measured baseline comparison.")
    ap.add_argument("--out", type=Path,
                    default=HERE / "comparison-2026-08-27" / "ablation-corrector-tools-waterfall.json")
    ap.add_argument("--include-extract-template", action="store_true",
                     help="add a final 8_extract_template_from_reaction tier -- NOT part of "
                          "fix_step's real chain (invents templates outside RXN1) and can hang "
                          "on pathological inputs, so each call is capped at "
                          f"{EXTRACT_TEMPLATE_TIMEOUT_S}s. Off by default.")
    ap.add_argument("--source", type=Path, default=ROOT / "data" / "synllama-raw-output.csv",
                     help="override the source CSV (same schema) -- e.g. a copy with "
                          "fix_invalid_json.py's repairs patched in")
    ap.add_argument("--include-analog-tools", action="store_true",
                     help="append fix_via_analogue_building_block and "
                          "fix_via_product_analogue_retro as two further tiers. Off by "
                          "default: they are explicit-method-only in fix_step, so they "
                          "are outside the strict chain the SynLlama comparison reports.")
    args = ap.parse_args()

    extra: list[str] = []
    if args.include_analog_tools:
        extra += ANALOG_TOOL_ORDER
    if args.include_extract_template:
        extra.append("extract_template_from_reaction")
    VARIANTS = _cumulative_variants(extra or None)

    source = args.source
    with source.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

    # Which of the 1,000 targets actually need correcting under this pool's
    # length cap (matches frozen_failures() in run_repair_frozen.py).
    needs_fix: dict[str, str] = {}
    for row in rows:
        try:
            route = _parse_route_json(row["response"])
            report = _validate_route_dict(route, analog_product_threshold=None)
            ok = report.all_building_blocks_valid and report.all_reactions_passed
        except Exception:
            ok = False
        if not ok and len(row["response"]) <= args.max_len:
            needs_fix[row["smiles"]] = row["response"]

    print(f"{len(rows)} targets total, {len(needs_fix)} need correcting under "
          f"--max-len {args.max_len}\n", flush=True)

    print("running variant: 0_before (no correction -- raw baseline) ...", flush=True)
    before_metrics = score_effective_dataset([(row["smiles"], row["response"]) for row in rows])
    before_metrics["routes_with_a_reconstruction_attempt"] = 0
    before_metrics["routes_needing_fix"] = len(needs_fix)
    results = {"0_before": before_metrics}
    print(f"  -> {before_metrics}\n", flush=True)

    ts = CorrectorToolset()
    for name, enabled in VARIANTS.items():
        print(f"running variant: {name} (tools: {sorted(enabled)}) ...", flush=True)
        effective = []
        fixed_count = 0
        for i, row in enumerate(rows, start=1):
            original = row["response"]
            if row["smiles"] in needs_fix:
                corrected = await build_corrected_route(
                    ts, original, enabled, target_smiles=row["smiles"],
                    ratio_cap=args.max_reactant_ratio,
                    min_reactants=args.min_reactants,
                )
                if corrected is not None:
                    fixed_count += 1
                    effective.append((row["smiles"], json.dumps(corrected)))
                else:
                    effective.append((row["smiles"], original))
            else:
                effective.append((row["smiles"], original))
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

        metrics = score_effective_dataset(effective)
        metrics["routes_with_a_reconstruction_attempt"] = fixed_count
        metrics["routes_needing_fix"] = len(needs_fix)
        results[name] = metrics
        print(f"  -> {metrics}\n", flush=True)

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print("=" * 78)
    print(f"{'metric':<28}", *[f"{k:>16}" for k in results], sep="")
    for metric in [
        "valid_json_percent", "template_mem_percent", "bb_selection_percent",
        "valid_smiles_percent", "matched_reactants_percent",
        "good_products_strict_percent",
    ]:
        row_vals = [f"{results[v][metric]:>16}" for v in results]
        print(f"{metric:<28}", *row_vals, sep="")
    print("=" * 78)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
