from __future__ import annotations

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

from .validation import agent as validation_agent
from .optimization import agent as optimization_agent
from .chemspace import agent as chemspace_agent
from .corrector import agent as corrector_agent

from ..chemspacetool import ChemspaceDeps
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

4. corrector agent
   - given a full route (same <SMILES>,<difficulty>,<JSON> input the validation agent
     takes) or a single reaction step, proposes corrected reactants for any invalid step
     and confirms each fix is actually valid before returning it. It also checks whether
     a fix to one step breaks the chain to other steps in the route (a step's corrected
     reactants no longer matching what an earlier step actually produces).

Your job is not to do all calculations yourself.
Your job is to decide which specialist agent should be called, call it,
then combine the results into a clear final report.

General workflow:
1. If the user gives a reaction pathway, call the validation agent first.
2. If the user asks for price or availability, call the ChemSpace agent.
3. If the user gives hazard codes or asks for hazard score, call the optimization agent.
4. If the validation agent reports any step as invalid and the user asks for a fix or
   correction, call the corrector agent with the ENTIRE original route input, exactly as
   the user gave it (the full <SMILES>,<difficulty>,<JSON> string) — do not extract or
   paraphrase individual reactants/products yourself, the corrector parses the whole
   route and every step itself.
5. When reporting the corrector's result, state for EVERY step whether it was already
   valid, fixed, or unfixable, and explicitly state whether the route is internally
   consistent as a whole (chain_consistent) — not just whether each step individually
   validates. If a step's fix orphaned another step (chain_consistent is false), say so
   and name which step is now disconnected, rather than declaring the route valid.
6. If the user asks for a full route evaluation, call validation, ChemSpace, and optimization,
   then summarize all outputs together.

Do not invent prices.
Do not invent ChemSpace results.
Do not invent hazard codes.
Do not invent reactant fixes — only report what the corrector agent returns.
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
    Call the corrector agent to propose and validate a fix for a reaction step
    that the validation agent reported as invalid.
    """
    subagent = _ensure_model(corrector_agent)
    result = await subagent.run(user_input)
    return str(result.output)


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


    
