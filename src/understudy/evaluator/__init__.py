"""Evaluator Agent: LLM-powered evaluation of AI agents.

The evaluator agent sits on top of understudy's primitives (Scene, run, check)
and adds an intelligence layer that can:
- Understand agents from code, docs, specs, and golden traces
- Generate comprehensive test scenarios
- Execute tests and analyze results deeply
- Iterate: drill down on failures, expand coverage
- Report what's broken and why
"""

from .agent import EvaluatorAgent
from .analyzer import DeepAnalyzer, FailurePattern
from .coverage import CoverageReport, CoverageTracker
from .generator import GenerationMode, SceneGenerator
from .inputs import AgentSpec, Capability, Constraint, InputProcessor
from .iteration import IterationController, IterationDecision
from .report import EvaluationReport, EvaluationSummary, FailureReport

__all__ = [
    "EvaluatorAgent",
    "AgentSpec",
    "Capability",
    "Constraint",
    "InputProcessor",
    "SceneGenerator",
    "GenerationMode",
    "DeepAnalyzer",
    "FailurePattern",
    "CoverageTracker",
    "CoverageReport",
    "IterationController",
    "IterationDecision",
    "EvaluationReport",
    "EvaluationSummary",
    "FailureReport",
]
