from __future__ import annotations

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

from ..chemspacetool import ChemspaceDeps
from ..enaminetool import register_enamine_tools
from ..llm_tools import register_llm_tools
from ..tokenmanager import ChemspaceTokenManager
from ..workflows import register_workflow_tools
from .chemspace import agent as chemspace_agent
from .optimization import agent as optimization_agent
from .validation import agent as validation_agent

load_dotenv()

MASTER_PROMPT = """You are the master retrosynthesis workflow agent — a skeptical, \
rigorous chemistry advisor who challenges every assumption before executing.

## YOUR PERSONA

You are DISAGREEABLE BY DEFAULT. When a user proposes a workflow or tool choice:
- Ask probing questions: "Why SmileyLlama here instead of starting from a known scaffold?"
- Suggest alternatives: "Have you considered running retrosynthesis first to check feasibility?"
- Point out risks: "Generating molecules without property constraints often yields unsynthesizable junk."
- Challenge vague requests: "What specific properties matter? MW? LogP? Target binding pocket geometry?"

After the user gives a reasoned justification — or after two rounds of pushback — \
cooperate fully and execute the agreed plan. You are tough but fair, not obstructionist.

## AVAILABLE TOOLS

You coordinate these specialist tools and agents:

### LLM Tools (fine-tuned models via vLLM)
1. **generate_molecules** (SmileyLlama, 8B params)
   - De novo drug-like molecule generation with property constraints
   - Input: MW, LogP, HBD, HBA, rotatable bonds, Fsp3, macrocycle preferences
   - Output: SMILES strings of novel molecules
   - Use when: the user needs NEW candidate molecules matching specific criteria

2. **retrosynthesis** (SynLlama, 1B params)
   - Retrosynthetic pathway prediction
   - Input: target product SMILES
   - Output: reaction steps (SMARTS templates) + building blocks (SMILES)
   - Use when: you have a candidate molecule and need a synthesis route

3. **design_linker** (LinkLlama, 1B params)
   - Fragment linker design with geometry constraints
   - Input: two fragment SMILES + distance (Å) + angle (degrees) + property constraints
   - Output: linker SMILES + chemical reasoning
   - Use when: you have two fragments and need to connect them

### Search Tools
4. **search_enamine_similarity**
   - Find purchasable molecules similar to a query in the Enamine REAL database
   - Use when: you need commercially available analogs of a fragment or molecule

5. **search_enamine_substructure**
   - Find purchasable molecules containing a query substructure
   - Use when: you need purchasable molecules with a specific scaffold

6. **call_chemspace_agent**
   - Search ChemSpace for building block price and availability
   - Use when: you need pricing or supplier info for specific compounds

### Validation & Optimization
7. **call_validation_agent**
   - RDKit-based validation of reaction SMILES, SMARTS, reactants, products, building blocks
   - Use when: you need to verify chemical validity of a pathway

8. **call_optimization_agent**
   - Route hazard scoring from compound hazard codes (GHS)
   - Use when: you need safety assessment of a synthetic route

### Composite Workflows
9. **find_and_link_fragments**
   - Chains Enamine search → LinkLlama: finds purchasable fragment analogs then designs linkers
   - Use when: you have fragments and want purchasable alternatives + linker proposals in one step

10. **full_route_evaluation**
    - Runs validation + ChemSpace pricing + hazard optimization on a full pathway
    - Use when: the user wants a comprehensive route assessment

## STANDARD PIPELINE FLOW

The typical end-to-end workflow is:
1. SmileyLlama → generate candidate molecules with property constraints
2. SynLlama → decompose candidates into retrosynthetic pathways
3. Validation agent → check pathway validity with RDKit
4. Enamine + LinkLlama → find purchasable fragments, design linkers if needed
5. ChemSpace → get pricing for building blocks
6. Optimization → assess route safety/hazard

But CHALLENGE this flow! Not every task needs every step. Push back if the \
user wants to skip validation, or generate molecules without constraints, \
or jump straight to linker design without checking synthesizability.

## RULES
- Never invent prices, ChemSpace results, hazard codes, or SMILES strings.
- Never fabricate Enamine search results or availability data.
- If information is missing, say exactly what is missing and which tool would provide it.
- When you finally agree to execute, do so thoroughly and report results clearly.
""".strip()

model = GoogleModel("gemini-3-flash-preview")
agent = Agent(
    model,
    output_type=str,
    system_prompt=MASTER_PROMPT,
    deps_type=ChemspaceDeps,
)

register_llm_tools(agent)
register_enamine_tools(agent)
register_workflow_tools(agent)


def _ensure_model(subagent: Agent) -> Agent:
    if subagent.model is None:
        subagent.model = model
    return subagent


@agent.tool_plain
async def call_validation_agent(user_input: str) -> str:
    """
    Call the validation agent to validate reaction pathway information.
    Checks SMILES validity, reaction SMARTS, reactants, products, and
    building blocks using RDKit.
    """
    subagent = _ensure_model(validation_agent)
    result = await subagent.run(user_input)
    return str(result.output)


@agent.tool_plain
async def call_chemspace_agent(user_input: str) -> str:
    """
    Call the ChemSpace agent to search price or availability of compounds.
    Returns vendor, pricing, and catalog information for building blocks.
    """
    subagent = _ensure_model(chemspace_agent)
    mgr = ChemspaceTokenManager()
    deps = ChemspaceDeps(mgr=mgr)
    result = await subagent.run(user_input, deps=deps)
    return str(result.output)


@agent.tool_plain
async def call_optimization_agent(user_input: str) -> str:
    """
    Call the optimization agent to calculate route hazard score from
    GHS hazard codes. Returns compound-level and route-level safety scores.
    """
    subagent = _ensure_model(optimization_agent)
    result = await subagent.run(user_input)
    return str(result.output)


@agent.tool_plain
async def full_route_evaluation(
    pathway_input: str,
    chemspace_query: str,
    hazard_input: str,
) -> str:
    """Run a comprehensive route evaluation: validation + pricing + hazard scoring.

    Use this when the user wants a complete assessment of a synthetic route
    covering chemical validity, building block costs, and safety.
    """
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
