import os
import warnings

from pydantic_ai_harness.experimental import HarnessExperimentalWarning

# hide experimental subagents warning
warnings.filterwarnings("ignore", category=HarnessExperimentalWarning)

# hide pydantic banner
os.environ["PYDANTIC_AI_NO_BANNER"] = "1"