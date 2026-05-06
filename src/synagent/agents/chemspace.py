import asyncio

from ..chemspacetool import (
    ChemspaceDeps,
    search_exact,
    search_substructure,
    search_similarity,
)
from ..tokenmanager import ChemspaceTokenManager
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

from dotenv import load_dotenv

load_dotenv()


agent = Agent(
    model = GoogleModel('gemini-3-flash-preview'),
    deps_type=ChemspaceDeps,
    tools=[search_exact, search_substructure, search_similarity],
)


async def main():
    mgr = ChemspaceTokenManager()
    deps = ChemspaceDeps(mgr=mgr)

    result = await agent.run(
        "Find exact ChemSpace matches for CCO",
        deps=deps,
    )
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())