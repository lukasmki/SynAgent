import os
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.experimental.subagents import SubAgents

from synagent.analogues import AnalogueSearch
from synagent.chemspace import Chemspace
from synagent.corrector import Corrector
from synagent.model_calls import ModelCalls
from synagent.prompts import DETERMINISTIC, DISAGREEABLE
from synagent.retrosynthesis import Retrosynthesis
from synagent.scoring import Scoring
from synagent.storage import Storage
from synagent.validation import SynthesisValidation

Provider = Literal["ollama", "anthropic", "mistral"]
Persona = Literal["deterministic", "disagreeable"]

_INSTRUCTIONS: dict[Persona, str] = {
    "deterministic": DETERMINISTIC,
    "disagreeable": DISAGREEABLE,
}


def _chemspace_caps() -> list:
    """Chemspace, but only when an API key is actually present.

    `ChemspaceToolset.__init__` constructs `ChemspaceAPI` eagerly, and that
    raises `ValueError` when `CHEMSPACE_API_KEY` is unset. corrector.py2
    commented Chemspace out of the top-level capability list for exactly this
    reason, but left it in both subagents — so `get_agent()` still raised for
    anyone without a key, making the agent impossible to construct.

    Returning an empty list keeps the agent constructible without a key while
    preserving full Chemspace behaviour for anyone who has one.
    """
    return [Chemspace()] if os.getenv("CHEMSPACE_API_KEY") else []


def build_model(model_name: str, provider: Provider = "ollama"):
    """Build the orchestrator model.

    The orchestrator is the LLM that drives the agent loop and decides which
    tools to call. It is a separate thing from the three fine-tuned chemistry
    models (SmileyLlama / SynLlama / LinkLlama), which are reached through the
    `model_calls` toolset and always need their own vLLM server.

    provider="anthropic" exists so the agent loop — capability wiring, tool
    gating, subagent routing — can be exercised without standing up Ollama.
    It does NOT remove the vLLM dependency for the chemistry models: Claude
    will not reproduce SynLlama's route format or SmileyLlama's property-
    conditioned distribution. Use it with mocked chemistry tools.
    """
    if provider == "anthropic":
        # Imported lazily so the anthropic extra stays optional.
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(model_name)

    if provider == "mistral":
        # Reads MISTRAL_API_KEY from the environment. Imported lazily so the
        # mistralai dependency stays optional.
        from pydantic_ai.models.mistral import MistralModel

        return MistralModel(model_name)

    # Ollama, via its OpenAI-compatible endpoint. The `think` settings are
    # Qwen3.5-specific and must not be sent to Anthropic.
    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        settings=ModelSettings(thinking=False, extra_body={"think": False}),
    )


def get_agent(
    model_name: str,
    provider: Provider = "ollama",
    persona: Persona = "deterministic",
) -> Agent[None, str]:
    """Build the full SynAgent.

    Args:
        model_name: orchestrator model id, e.g. "qwen3.5" or
            "claude-sonnet-5".
        provider: "ollama" (default) or "anthropic".
        persona: "deterministic" (default) for reproducible tool sequences,
            which is what the pipeline tests rely on; "disagreeable" for the
            interactive advisor that argues before executing. Do not use
            "disagreeable" in automated tests — it will spend turns pushing
            back instead of calling tools.
    """
    model = build_model(model_name, provider)

    agent = Agent(
        model,
        instructions=_INSTRUCTIONS[persona],
        capabilities=[
            AnalogueSearch(),
            # Chemspace(),  # kept disabled from corrector.py2 — needs CHEMSPACE_API_KEY
            ModelCalls(),
            SynthesisValidation(),
            Corrector(),
            Retrosynthesis(),
            Scoring(),
            Storage(),
            SubAgents(
                agents={
                    "analogue": Agent(
                        model,
                        description="Can access the local building block and reactions database as well as the Chemspace API.",
                        instructions="You are a molecule/reaction analogue sub-agent.",
                        capabilities=[AnalogueSearch(), *_chemspace_caps()],
                    ),
                    "worker": Agent(
                        model,
                        description="General purpose sub-agent.",
                        instructions="You are a sub-agent.",
                        capabilities=[
                            *_chemspace_caps(),
                            SynthesisValidation(),
                            AnalogueSearch(),
                        ],
                    ),
                },
                shared_capabilities=[],
            ),
            # CodeMode(tools={"code_mode": True}),
        ],
    )
    return agent
