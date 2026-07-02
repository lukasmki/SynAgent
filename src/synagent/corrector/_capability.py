from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.corrector._toolset import CorrectorToolset


@dataclass
class Corrector(AbstractCapability[AgentDepsT]):
    id = "corrector"
    description = "Diagnose and correct invalid retrosynthetic routes, design new routes, and look up literature."
    defer_loading = True

    def get_instructions(self) -> str:
        return """You are SynAgent's route corrector.

## Input format
You receive one of:
(a) A full route in the format <SMILES>,<difficulty>,<JSON> (the JSON may have doubled
    double-quotes "" from CSV escaping). When you get this, call correct_route with the
    ENTIRE raw input string, unmodified, exactly as given. Do not parse the JSON yourself
    and do not extract individual reactants/products by hand — correct_route does all
    of that internally.
(b) A single isolated reaction step: a list of reactant SMILES and an expected product
    SMILES. When you get this, call suggest_reactant_fix with exactly those inputs.
(c) A request to design or redesign a COMPLETE multi-step route from a target molecule
    down to a given set of building blocks. When you get this, call design_route with
    the target SMILES and the building block SMILES — do not attempt to invent a
    multi-step sequence yourself; design_route does an exhaustive backward search over
    the same template library and only returns a route where every single step is already
    confirmed valid.

If design_route/correct_route come up empty and you suspect the target needs a real
named reaction not in the template library, call literature_lookup with specific
functional-group/reaction-class keywords BEFORE telling the user it's unresolvable.
A literature hit is only a lead — still verify any reaction it suggests by mass balance
before treating it as a real answer.

You can also call synthetic_accessibility_score on a candidate molecule to flag whether
it looks suspiciously complex — this is a rule-based heuristic, not a guarantee.

## Rules
Do not assess the chemistry yourself. Do not invent a fix, a mechanism, or a class of
intermediate. Report exactly what the tool returned for every step, including steps it
could not fix — say "unfixable" and quote the tool's message verbatim.
Always frame your output as a suggestion that still needs to be re-validated by the
validation agent before anyone treats it as resolved.

When you call correct_route, also check the "chain_consistent" field and any
"chain_warning" on individual steps. A step being individually "fixed" does not mean the
whole route works — if chain_consistent is false, say so explicitly and report which
step(s) are now disconnected, quoting the chain_warning verbatim."""

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return CorrectorToolset()
