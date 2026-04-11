"""Core data models for agentic flow evaluation."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from ..trace import ToolCall
from ..validation import SceneValidationError, format_pydantic_error


class Task(BaseModel):
    """A task for an agentic agent to complete."""

    description: str
    goal: str
    constraints: list[str] = Field(default_factory=list)
    initial_state: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 100
    max_tokens: int | None = None


class Artifact(BaseModel):
    """An output artifact produced by an agent."""

    name: str
    artifact_type: str
    content: Any = None
    path: str | None = None
    timestamp: datetime | None = None


class Step(BaseModel):
    """A single step in an agentic execution."""

    step_number: int
    step_type: str
    reasoning: str | None = None
    action: str | None = None
    action_args: dict[str, Any] = Field(default_factory=dict)
    observation: Any = None
    error: str | None = None
    tokens_used: int = 0
    latency_ms: int = 0
    timestamp: datetime | None = None


class AgenticExpectations(BaseModel):
    """Expectations for an agentic execution."""

    goal_predicate: str | None = None
    golden_output: dict[str, Any] | None = None
    max_steps: int | None = None
    max_tokens: int | None = None
    max_retries: int | None = None
    required_actions: list[str] = Field(default_factory=list)
    forbidden_actions: list[str] = Field(default_factory=list)
    reasoning_rubrics: list[str] = Field(default_factory=list)


class AgenticTrace(BaseModel):
    """The full execution trace of an agentic flow.

    Parallel to Trace but for autonomous agent flows.
    """

    scene_id: str
    task: Task
    steps: list[Step] = Field(default_factory=list)
    outcome: str = "pending"
    final_state: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.steps)

    @property
    def total_latency_ms(self) -> int:
        return sum(s.latency_ms for s in self.steps)

    @property
    def duration(self) -> timedelta | None:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return None

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Compatibility property for shared tooling."""
        calls = []
        for step in self.steps:
            if step.step_type == "act" and step.action:
                calls.append(
                    ToolCall(
                        tool_name=step.action,
                        arguments=step.action_args,
                        result=step.observation,
                        timestamp=step.timestamp,
                        error=step.error,
                    )
                )
        return calls

    def action_sequence(self) -> list[str]:
        """Ordered list of action names taken."""
        return [s.action for s in self.steps if s.step_type == "act" and s.action]

    def actions_to(self, action_name: str) -> list[Step]:
        """Get all steps for a specific action."""
        return [s for s in self.steps if s.step_type == "act" and s.action == action_name]

    def performed(self, action_name: str, **kwargs: Any) -> bool:
        """Check if an action was performed, optionally with specific args."""
        for step in self.steps:
            if step.step_type == "act" and step.action == action_name:
                if not kwargs:
                    return True
                if all(step.action_args.get(k) == v for k, v in kwargs.items()):
                    return True
        return False

    def thinking_steps(self) -> list[Step]:
        """Get all thinking/reasoning steps."""
        return [s for s in self.steps if s.step_type == "think"]

    def observation_steps(self) -> list[Step]:
        """Get all observation steps."""
        return [s for s in self.steps if s.step_type == "observe"]

    def retry_count(self) -> int:
        """Count the number of retried actions (actions with errors)."""
        return sum(1 for s in self.steps if s.step_type == "act" and s.error)

    def conversation_text(self) -> str:
        """Render the execution as readable text (for judge input)."""
        lines = [f"TASK: {self.task.description}", f"GOAL: {self.task.goal}", ""]
        for step in self.steps:
            if step.step_type == "think":
                lines.append(f"[THINK]: {step.reasoning}")
            elif step.step_type == "act":
                lines.append(f"[ACT]: {step.action}({step.action_args})")
                if step.observation is not None:
                    obs_str = str(step.observation)
                    if len(obs_str) > 200:
                        obs_str = obs_str[:200] + "..."
                    lines.append(f"  <- {obs_str}")
                if step.error:
                    lines.append(f"  [ERROR]: {step.error}")
            elif step.step_type == "observe":
                obs_str = str(step.observation)
                if len(obs_str) > 200:
                    obs_str = obs_str[:200] + "..."
                lines.append(f"[OBSERVE]: {obs_str}")
        lines.append("")
        lines.append(f"OUTCOME: {self.outcome}")
        return "\n".join(lines)

    @classmethod
    def from_json(cls, data: dict) -> "AgenticTrace":
        """Create an AgenticTrace from JSON data."""
        task_data = data.get("task", {})
        task = Task(**task_data)

        steps = []
        for step_data in data.get("steps", []):
            if "timestamp" in step_data and isinstance(step_data["timestamp"], str):
                step_data["timestamp"] = datetime.fromisoformat(step_data["timestamp"])
            steps.append(Step(**step_data))

        artifacts = []
        for artifact_data in data.get("artifacts", []):
            if "timestamp" in artifact_data and isinstance(artifact_data["timestamp"], str):
                artifact_data["timestamp"] = datetime.fromisoformat(artifact_data["timestamp"])
            artifacts.append(Artifact(**artifact_data))

        started_at = data.get("started_at")
        if started_at and isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)

        finished_at = data.get("finished_at")
        if finished_at and isinstance(finished_at, str):
            finished_at = datetime.fromisoformat(finished_at)

        return cls(
            scene_id=data.get("scene_id", "unknown"),
            task=task,
            steps=steps,
            outcome=data.get("outcome", "pending"),
            final_state=data.get("final_state", {}),
            artifacts=artifacts,
            started_at=started_at,
            finished_at=finished_at,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "AgenticTrace":
        """Load an AgenticTrace from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_json(data)


class AgenticScene(BaseModel):
    """A scene for agentic flow evaluation.

    Parallel to Scene but for autonomous agent flows.
    """

    id: str
    description: str = ""
    task: Task
    environment: dict[str, Any] = Field(default_factory=dict)
    expectations: AgenticExpectations = Field(default_factory=AgenticExpectations)

    @classmethod
    def from_file(cls, path: str | Path) -> "AgenticScene":
        """Load a scene from a YAML or JSON file."""
        path = Path(path)

        try:
            with open(path) as f:
                data = yaml.safe_load(f) if path.suffix in (".yaml", ".yml") else json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Scene file not found: {path}") from e
        except yaml.YAMLError as e:
            raise SceneValidationError(
                f"Invalid YAML syntax in '{path}':\n  {e}", file_path=path
            ) from e

        if data is None:
            raise SceneValidationError(f"Scene file is empty: {path}", file_path=path)

        try:
            return cls._from_dict(data)
        except ValidationError as e:
            raise SceneValidationError(
                format_pydantic_error(e, file_path=path, data=data), file_path=path
            ) from e

    @classmethod
    def _from_dict(cls, data: dict) -> "AgenticScene":
        """Parse a scene dict."""
        task_raw = data.get("task")
        if isinstance(task_raw, dict):
            data["task"] = Task(**task_raw)

        expectations_raw = data.get("expectations")
        if isinstance(expectations_raw, dict):
            data["expectations"] = AgenticExpectations(**expectations_raw)

        return cls(**data)
