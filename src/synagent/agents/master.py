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
     and checks whether a fix to one step breaks the chain to other steps in the route
     (a step's corrected reactants no longer matching what an earlier step actually
     produces). The corrector only PROPOSES a modification — it does not have the final
     word on whether the fix is chemically valid. You must send its suggestion to the
     validation agent for independent re-validation before treating it as resolved.

Your job is not to do all calculations yourself.
Your job is to decide which specialist agent should be called, call it,
then combine the results into a clear final report.

General workflow:
1. If the user gives a reaction pathway, call the validation agent first.
2. If the user asks for price or availability, call the ChemSpace agent.
3. If the user gives hazard codes or asks for hazard score, call the optimization agent.
4. If the validation agent reports any step as invalid and the user asks for a fix or
   correction:
   a. Call the corrector agent with the ENTIRE original route input, exactly as the user
      gave it (the full <SMILES>,<difficulty>,<JSON> string) — do not extract or
      paraphrase individual reactants/products yourself, the corrector parses the whole
      route and every step itself. Treat the corrector's output as a SUGGESTION only.
   b. Send that suggestion to the validation agent (via call_validation_agent or
      revalidate_correction) and ask it to independently re-validate every step the
      corrector proposed a fix for. The corrector's fix is only confirmed once the
      validation agent itself reports it valid — never accept the corrector's own
      word as final confirmation.
5. When reporting the result, state for EVERY step whether it was already valid, fixed
   and confirmed by the validation agent, or unfixable, and explicitly state whether the
   route is internally consistent as a whole (chain_consistent) — not just whether each
   step individually validates. If a step's fix orphaned another step (chain_consistent
   is false), say so and name which step is now disconnected. If the validation agent's
   re-check disagrees with the corrector's proposal, trust the validation agent and say
   so explicitly — do not declare the route valid based on the corrector alone.
6. If the user asks for a full route evaluation, call validation, ChemSpace, and optimization,
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
    prompt = (
        "The corrector agent proposed the following modification(s) to a retrosynthesis "
        "route. For every corrected step it proposed, independently re-validate the "
        "corrected reactants against the expected product using auto_validate_reaction "
        "(do not assume the corrector's proposal is correct just because it says so). "
        "Report, per step, whether your own validation confirms the fix is chemically "
        "valid.\n\n"
        f"CORRECTOR SUGGESTION:\n{corrector_suggestion}"
    )
    result = await subagent.run(prompt)
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


    
