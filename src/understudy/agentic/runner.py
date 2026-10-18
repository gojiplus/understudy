"""Runner: orchestrates agentic flow execution."""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ..mocks import MockToolkit
from .models import AgenticScene, AgenticTrace, Step, Task

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from a single agent step."""

    step_type: str
    reasoning: str | None = None
    action: str | None = None
    action_args: dict[str, Any] | None = None
    observation: Any = None
    error: str | None = None
    tokens_used: int = 0


class AgenticApp(Protocol):
    """Protocol for agentic applications that understudy can drive.

    Implementations wrap the actual agent framework and expose
    a step-based interface for autonomous execution.
    """

    def start(
        self,
        task: Task,
        environment: dict[str, Any] | None = None,
        *,
        mocks: MockToolkit | None = None,
    ) -> None:
        """Initialize the agent with a task and environment."""
        ...

    def step(self) -> StepResult:
        """Execute one step and return the result."""
        ...

    def is_done(self) -> bool:
        """Check if the agent has completed or should stop."""
        ...

    def get_outcome(self) -> str:
        """Get the final outcome status."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Get the current state of the agent."""
        ...

    def stop(self) -> None:
        """Clean up and stop the agent."""
        ...


def run_agentic(
    app: AgenticApp,
    scene: AgenticScene,
    mocks: MockToolkit | None = None,
) -> AgenticTrace:
    """Run an agentic scene and return the trace.

    Args:
        app: The agentic application to test.
        scene: The agentic scene (task fixture) to run.
        mocks: Optional mock toolkit for tool responses.

    Returns:
        An AgenticTrace recording everything that happened.

    Raises:
        ValueError: If mocks are supplied to an adapter without interception.
    """
    trace = AgenticTrace(
        scene_id=scene.id,
        task=scene.task,
        started_at=datetime.now(UTC),
    )

    logger.info("Running agentic scene: %s", scene.id)

    if mocks is not None and not getattr(app, "supports_tool_mocking", False):
        raise ValueError(
            "Agentic adapter must intercept tools before execution to accept mocks"
        )

    try:
        if mocks is None:
            app.start(task=scene.task, environment=scene.environment)
        else:
            app.start(task=scene.task, environment=scene.environment, mocks=mocks)
        step_number = 0
        max_steps = scene.task.max_steps
        max_tokens = scene.task.max_tokens
        total_tokens = 0

        while not app.is_done():
            step_number += 1

            if step_number > max_steps:
                trace.outcome = "max_steps_exceeded"
                logger.warning("Scene %s: max steps exceeded (%d)", scene.id, max_steps)
                break

            start_time = time.perf_counter()
            result = app.step()
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            step = Step(
                step_number=step_number,
                step_type=result.step_type,
                reasoning=result.reasoning,
                action=result.action,
                action_args=result.action_args or {},
                observation=result.observation,
                error=result.error,
                tokens_used=result.tokens_used,
                latency_ms=latency_ms,
                timestamp=datetime.now(UTC),
            )
            trace.steps.append(step)

            total_tokens += result.tokens_used
            if max_tokens and total_tokens > max_tokens:
                trace.outcome = "max_tokens_exceeded"
                logger.warning(
                    "Scene %s: max tokens exceeded (%d)", scene.id, max_tokens
                )
                break

            logger.debug(
                "Step %d: %s - %s", step_number, result.step_type, result.action or ""
            )

        if trace.outcome == "pending":
            trace.outcome = app.get_outcome()

        trace.final_state = app.get_state()

    except Exception as e:
        trace.outcome = "error"
        trace.metadata["error"] = str(e)
        logger.error("Scene %s: error - %s", scene.id, e)

    finally:
        app.stop()
        trace.finished_at = datetime.now(UTC)

    logger.info("Scene %s finished: %s", scene.id, trace.outcome)
    return trace
