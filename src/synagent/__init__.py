import os
import warnings

from pydantic_ai_harness.experimental import HarnessExperimentalWarning

# hide experimental subagents warning
warnings.filterwarnings("ignore", category=HarnessExperimentalWarning)

# hide pydantic banner
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

# `synagent.api` loads the whole agent stack, so it is imported on first use:
# importing a submodule such as `synagent.tracing` doesn't pay for it.
_API_NAMES = (
    "Chat",
    "Reply",
    "Report",
    "Route",
    "SynAgent",
    "ToolCall",
    "ToolCalls",
    "specs",
)

__all__ = list(_API_NAMES)


def __getattr__(name: str):
    if name in _API_NAMES:
        from synagent import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
