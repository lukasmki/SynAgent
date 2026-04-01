from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from synagent.models import BuildingBlockResult, ReactionResult, SynLlamaReport,OptimizationReport
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

OPTIMIZER_PROMPT = """
        You are a retrosynthesis optimizer.
        You receive a SynLlamaReport.
        Only optimize based on building block price and hazard information.
        Use DuckDuckGo search only to look up building block price and hazard information.
        Do not evaluate reaction yield.
        If a building block is invalid, flag it rather than searching deeply.
        Summarize evidence conservatively and return a structured SearchResult""".strip()



agent = Agent(
    system_prompt=OPTIMIZER_PROMPT,
    tools = [duckduckgo_search_tool(max_results=5)],
    output_type=OptimizationReport,
)

@agent.tool_plain
async def create_report(report: SynLlamaReport) -> OptimizationReport:
    return report

"""@optimizer.tool_plain
async def optimize(task: str) -> str:
    result = await optimizer.run(task)
    return result.output"""



