"""understudy: simulation and trace-based evaluation for agentic systems.

The simulated user is an understudy standing in for a real user.
You write scenes, run rehearsals, and check the performance —
not by reading the script, but by inspecting what actually happened.
"""

from importlib.metadata import version

from . import agentic
from .agentic import (
    AgenticApp,
    AgenticCheckItem,
    AgenticCheckResult,
    AgenticExpectations,
    AgenticMetricResult,
    AgenticScene,
    AgenticTrace,
    Artifact,
    Step,
    StepResult,
    Task,
    check_agentic,
    run_agentic,
)
from .check import (
    CheckItem,
    CheckResult,
    EvaluationResult,
    check,
    evaluate,
    evaluate_batch,
)
from .compare import ComparisonResult, SceneComparison, compare_runs
from .diff import ToolCallDiff, TraceDiff, diff_tool_sequences, diff_traces
from .judge_backends import CallbackBackend, JudgeBackend, LiteLLMBackend
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
from .pytest_plugin import AssertionHelpers
from .replay import ReplayResult, create_replay_scene, load_trace, replay
from .runner import AgentApp, AgentResponse, run, simulate, simulate_batch
from .simulator import Simulator, SimulatorBackend
from .storage import EvaluationStorage, RunStorage, TraceStorage
from .suite import SceneResult, Suite, SuiteResults
from .trace import (
    AgentTransfer,
    StateSnapshot,
    ToolCall,
    Trace,
    TraceMetrics,
    Turn,
    TurnMetrics,
)
from .validation import SceneValidationError

__version__ = version("understudy")

__all__ = [
    "ADVERSARIAL_ROBUSTNESS",
    "FACTUAL_GROUNDING",
    "INSTRUCTION_FOLLOWING",
    "POLICY_COMPLIANCE",
    "TASK_COMPLETION",
    "TONE_EMPATHY",
    # rubrics
    "TOOL_USAGE_CORRECTNESS",
    "AgentApp",
    "AgentResponse",
    "AgentTransfer",
    "AgenticApp",
    "AgenticCheckItem",
    "AgenticCheckResult",
    "AgenticExpectations",
    "AgenticMetricResult",
    "AgenticScene",
    "AgenticTrace",
    "Artifact",
    # pytest
    "AssertionHelpers",
    "CallbackBackend",
    "CheckItem",
    "CheckResult",
    "ComparisonResult",
    "EvaluationResult",
    "EvaluationStorage",
    "Expectations",
    "FailureAnalysis",
    "FailureAnalyzer",
    # judges
    "Judge",
    # judge backends
    "JudgeBackend",
    "JudgeResult",
    "LiteLLMBackend",
    # metrics
    "MetricRegistry",
    "MetricResult",
    # mocks
    "MockToolkit",
    "Persona",
    "PersonaPreset",
    "ReplayResult",
    # storage
    "RunStorage",
    # models
    "Scene",
    "SceneComparison",
    "SceneResult",
    # validation
    "SceneValidationError",
    # simulator
    "Simulator",
    "SimulatorBackend",
    "StateSnapshot",
    "Step",
    "StepResult",
    # suite
    "Suite",
    "SuiteResults",
    "Task",
    "ToolCall",
    "ToolCallDiff",
    "ToolError",
    # trace
    "Trace",
    "TraceDiff",
    "TraceMetrics",
    "TraceStorage",
    "Turn",
    "TurnMetrics",
    # agentic module
    "agentic",
    # check / evaluate
    "check",
    "check_agentic",
    # compare
    "compare_runs",
    "create_replay_scene",
    "diff_tool_sequences",
    # diff
    "diff_traces",
    "evaluate",
    "evaluate_batch",
    "load_trace",
    # replay
    "replay",
    # runner / simulate
    "run",
    "run_agentic",
    "simulate",
    "simulate_batch",
]
