"""understudy: simulation and trace-based evaluation for agentic systems.

The simulated user is an understudy standing in for a real user.
You write scenes, run rehearsals, and check the performance —
not by reading the script, but by inspecting what actually happened.
"""

from .check import CheckItem, CheckResult, check
from .judges import Judge, JudgeResult
from .mocks import MockToolkit, ToolError
from .models import Expectations, Persona, PersonaPreset, Scene
from .prompts import (
    ADVERSARIAL_ROBUSTNESS,
    FACTUAL_GROUNDING,
    INSTRUCTION_FOLLOWING,
    POLICY_COMPLIANCE,
    TASK_COMPLETION,
    TONE_EMPATHY,
    TOOL_USAGE_CORRECTNESS,
)
from .runner import AgentApp, AgentResponse, run
from .storage import RunStorage
from .suite import SceneResult, Suite, SuiteResults
from .trace import AgentTransfer, ToolCall, Trace, Turn

__version__ = "0.1.0"

__all__ = [
    # models
    "Scene",
    "Persona",
    "PersonaPreset",
    "Expectations",
    # trace
    "Trace",
    "Turn",
    "ToolCall",
    "AgentTransfer",
    # runner
    "run",
    "AgentApp",
    "AgentResponse",
    # check
    "check",
    "CheckResult",
    "CheckItem",
    # suite
    "Suite",
    "SuiteResults",
    "SceneResult",
    # storage
    "RunStorage",
    # judges
    "Judge",
    "JudgeResult",
    # mocks
    "MockToolkit",
    "ToolError",
    # rubrics
    "TOOL_USAGE_CORRECTNESS",
    "POLICY_COMPLIANCE",
    "TONE_EMPATHY",
    "ADVERSARIAL_ROBUSTNESS",
    "TASK_COMPLETION",
    "FACTUAL_GROUNDING",
    "INSTRUCTION_FOLLOWING",
]
