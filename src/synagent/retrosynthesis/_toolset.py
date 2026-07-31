import json
from collections import deque
from pathlib import Path

from pydantic_ai import FunctionToolset
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

from synagent.validation._smarts import _COMMON_SMARTS

RDLogger.DisableLog("rdApp.*")

_MOLDB_PATH = Path(__file__).parent.parent / "analogues" / "data" / "building_blocks.h5"


def _get_target_from_report(messages: list) -> str | None:
    from synagent.validation._models import ValidationReport
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name != "validate_route":
                continue
            content = part.content
            try:
                if isinstance(content, ValidationReport):
                    return content.target_molecule
                if isinstance(content, dict):
                    return ValidationReport.model_validate(content).target_molecule
                if isinstance(content, str):
                    return ValidationReport.model_validate_json(content).target_molecule
                return ValidationReport.model_validate(
                    json.loads(json.dumps(content, default=str))
                ).target_molecule
            except Exception:
                pass
    return None


def _build_retro_rxns() -> list[tuple[str, rdChemReactions.ChemicalReaction]]:
    """Parse all _COMMON_SMARTS templates in reverse once at import time."""
    retro = []
    for fwd in _COMMON_SMARTS:
        rev = ">>".join(fwd.split(">>")[::-1])
        try:
            rxn = rdChemReactions.ReactionFromSmarts(rev)
            if rxn:
                rxn.Initialize()
                retro.append((fwd, rxn))
        except Exception:
            pass
    return retro


_RETRO_RXNS = _build_retro_rxns()


def _retro_bfs(
    target_smiles: str,
    engine,
    bb_threshold: float,
    max_depth: int,
    max_routes: int,
    max_nodes: int = 400,
) -> list[dict]:
    """BFS retrosynthesis using reversed _COMMON_SMARTS templates.

    Returns a list of complete routes, each route being a list of steps
    where all leaf reactants are in the building block database.
    """

    def is_bb(smi: str) -> bool:
        if engine is None:
            return False
        try:
            hits = engine.similarity(
                smi, bb_threshold, metric="cosine", n_workers=1, mol_format="smiles"
            )
            return len(hits) > 0
        except Exception:
            return False

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return []

    if is_bb(target_smiles):
        return [{"steps": [], "building_blocks": [target_smiles], "n_steps": 0}]

    # Each queue entry: (molecules_still_to_disconnect: frozenset, steps_so_far: list)
    queue: deque = deque()
    queue.append((frozenset([target_smiles]), []))
    complete_routes: list[dict] = []
    seen: set = set()
    nodes = 0

    while queue and len(complete_routes) < max_routes and nodes < max_nodes:
        to_expand, steps = queue.popleft()
        nodes += 1

        if len(steps) >= max_depth:
            continue

        # Always disconnect the longest (most complex) SMILES first
        ordered = sorted(to_expand, key=len, reverse=True)
        smiles = ordered[0]
        remaining = frozenset(ordered[1:])

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        for fwd_template, rxn in _RETRO_RXNS:
            try:
                outputs_list = rxn.RunReactants((mol,))
            except Exception:
                continue

            seen_precursor_sets: set = set()
            for outputs in outputs_list:
                precursors = []
                ok = True
                for m in outputs:
                    try:
                        Chem.SanitizeMol(m)
                        canon = Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
                        if canon:
                            precursors.append(canon)
                        else:
                            ok = False
                            break
                    except Exception:
                        ok = False
                        break

                if not ok or not precursors:
                    continue

                pkey = frozenset(precursors)
                if pkey in seen_precursor_sets:
                    continue
                seen_precursor_sets.add(pkey)

                non_bb = frozenset(p for p in precursors if not is_bb(p))
                new_step = {
                    "product": smiles,
                    "reactants": precursors,
                    "template": fwd_template,
                    "available_reactants": [p for p in precursors if p not in non_bb],
                }
                new_steps = steps + [new_step]
                new_to_expand = non_bb | remaining

                state_key = (new_to_expand, len(new_steps))
                if state_key in seen:
                    continue
                seen.add(state_key)

                if not new_to_expand:
                    all_bbs = sorted({p for s in new_steps for p in s["available_reactants"]})
                    complete_routes.append({
                        "n_steps": len(new_steps),
                        "building_blocks": all_bbs,
                        "steps": [
                            {
                                "step": i + 1,
                                "product": s["product"],
                                "reactants": s["reactants"],
                                "template": s["template"],
                            }
                            for i, s in enumerate(reversed(new_steps))
                        ],
                    })
                else:
                    queue.append((new_to_expand, new_steps))

    return sorted(complete_routes, key=lambda r: r["n_steps"])


class RetrosynthesisToolset(FunctionToolset[AgentDepsT]):
    include_return_schema = True

    def __init__(self):
        super().__init__()
        self.add_function(self.retro_search, name="retro_search")

    async def retro_search(
        self,
        ctx: RunContext[AgentDepsT],
        target_smiles: str = "from_report",
        max_depth: int = 4,
        max_routes: int = 3,
        bb_threshold: float = 0.9,
    ) -> dict:
        """Find new synthetic routes to a target molecule using retrosynthetic analysis.

        Applies the reaction template library in reverse (retrosynthetic direction) via
        BFS to find sequences of commercially available building blocks that can produce
        the target. Reads the target automatically from the last ValidationReport —
        no SMILES copying needed.

        Args:
            target_smiles: Target SMILES. Use "from_report" (default) to read automatically.
            max_depth: Maximum number of retrosynthetic steps (default 4).
            max_routes: Maximum number of complete routes to return (default 3).
            bb_threshold: Similarity threshold for building block availability check (default 0.9).

        Returns:
            dict with found routes, each listing steps, templates, and building blocks.
        """
        if (
            not target_smiles
            or len(target_smiles) < 5
            or target_smiles.strip().lower() in ("from_report", "from report", "auto", "")
        ):
            target_smiles = _get_target_from_report(ctx.messages) or ""

        if not target_smiles:
            return {"error": "No target SMILES found. Run validate_route first or provide target_smiles."}

        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            return {"error": f"Invalid target SMILES: {target_smiles}"}

        try:
            from FPSim2.FPSim2 import FPSim2Engine
            engine = FPSim2Engine(str(_MOLDB_PATH), in_memory_fps=True) if _MOLDB_PATH.exists() else None
        except ImportError:
            engine = None

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            routes = await loop.run_in_executor(
                None, _retro_bfs, target_smiles, engine, bb_threshold, max_depth, max_routes
            )
        except Exception as e:
            return {"error": f"Retrosynthesis search failed: {e}"}

        if not routes:
            return {
                "target": target_smiles,
                "routes_found": 0,
                "routes": [],
                "message": (
                    f"No complete routes found within {max_depth} steps using the "
                    f"{len(_RETRO_RXNS)}-template library. Try increasing max_depth or "
                    "lowering bb_threshold."
                ),
            }

        return {
            "target": target_smiles,
            "routes_found": len(routes),
            "routes": routes,
            "message": (
                f"Found {len(routes)} route(s). Shortest: {routes[0]['n_steps']} step(s) "
                f"from {len(routes[0]['building_blocks'])} building block(s)."
            ),
        }
