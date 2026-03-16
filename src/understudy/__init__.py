"""understudy: simulation and trace-based evaluation for agentic systems.

The simulated user is an understudy standing in for a real user.
You write scenes, run rehearsals, and check the performance —
not by reading the script, but by inspecting what actually happened.
"""

from importlib.metadata import version

from .check import CheckItem, CheckResult, EvaluationResult, check, evaluate, evaluate_batch
from .compare import ComparisonResult, SceneComparison, compare_runs
from .judges import FailureAnalysis, FailureAnalyzer, Judge, JudgeResult
from .metrics import MetricRegistry, MetricResult
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
from .runner import AgentApp, AgentResponse, LiteLLMBackend, run, simulate, simulate_batch
from .simulator import Simulator, SimulatorBackend
from .storage import EvaluationStorage, RunStorage, TraceStorage
from .suite import SceneResult, Suite, SuiteResults
from .trace import AgentTransfer, StateSnapshot, ToolCall, Trace, TraceMetrics, Turn, TurnMetrics

__version__ = version("understudy")

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
    "TraceMetrics",
    "TurnMetrics",
    "StateSnapshot",
    # metrics
    "MetricRegistry",
    "MetricResult",
    # runner / simulate
    "run",
    "simulate",
    "simulate_batch",
    "AgentApp",
    "AgentResponse",
    "LiteLLMBackend",
    # check / evaluate
    "check",
    "evaluate",
    "evaluate_batch",
    "CheckResult",
    "CheckItem",
    "EvaluationResult",
    # suite
    "Suite",
    "SuiteResults",
    "SceneResult",
    # storage
    "RunStorage",
    "TraceStorage",
    "EvaluationStorage",
    # compare
    "compare_runs",
    "ComparisonResult",
    "SceneComparison",
    # judges
    "Judge",
    "JudgeResult",
    "FailureAnalyzer",
    "FailureAnalysis",
    # mocks
    "MockToolkit",
    "ToolError",
    # simulator
    "Simulator",
    "SimulatorBackend",
    # rubrics
    "TOOL_USAGE_CORRECTNESS",
    "POLICY_COMPLIANCE",
    "TONE_EMPATHY",
    "ADVERSARIAL_ROBUSTNESS",
    "TASK_COMPLETION",
    "FACTUAL_GROUNDING",
    "INSTRUCTION_FOLLOWING",
]
