"""Session management for the HTTP simulator API."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..models import Expectations, Persona, Scene
from ..trace import ToolCall, Trace, Turn
from .models import SceneInput, ToolCallInput
from .ui_simulator import SimulatorBackend, UISimulator


@dataclass
class Session:
    """A simulation session."""

    id: str
    scene: Scene
    simulator: UISimulator
    trace: Trace
    turn_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    finished: bool = False
    finish_reason: str | None = None

    def add_agent_turn(
        self,
        content: str,
        tool_calls: list[ToolCallInput] | None = None,
    ) -> None:
        """Record an agent turn from the UI."""
        tc_list = []
        if tool_calls:
            for tc in tool_calls:
                tc_list.append(
                    ToolCall(
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        result=tc.result,
                    )
                )
        self.trace.turns.append(
            Turn(
                role="agent",
                content=content,
                tool_calls=tc_list,
                timestamp=datetime.now(),
            )
        )

    def add_user_turn(self, content: str) -> None:
        """Record a user turn (action taken)."""
        self.trace.turns.append(
            Turn(
                role="user",
                content=content,
                timestamp=datetime.now(),
            )
        )
        self.turn_count += 1

    def mark_finished(self, reason: str = "finished") -> None:
        """Mark the session as finished."""
        self.finished = True
        self.finish_reason = reason
        self.trace.finished_at = datetime.now()
        self.trace.terminal_state = reason


class SessionManager:
    """Manages simulation sessions."""

    def __init__(self, default_model: str = "gpt-4o"):
        self.sessions: dict[str, Session] = {}
        self.default_model = default_model
        self._backend_factory: type | None = None

    def set_backend_factory(self, factory: type) -> None:
        """Set the backend factory for creating simulator backends."""
        self._backend_factory = factory

    def _create_backend(self, model: str) -> SimulatorBackend:
        """Create a simulator backend."""
        if self._backend_factory:
            return self._backend_factory(model)

        from litellm import completion

        class LiteLLMBackend:
            def __init__(self, model: str):
                self.model = model

            def generate(self, prompt: str) -> str:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                )
                content = response.choices[0].message.content
                return content if content else ""

        return LiteLLMBackend(model)

    def create_session(
        self,
        scene_input: SceneInput,
        model: str | None = None,
    ) -> Session:
        """Create a new simulation session."""
        session_id = str(uuid.uuid4())
        model = model or self.default_model

        if isinstance(scene_input.persona, str):
            persona = Persona.from_preset(scene_input.persona)
        else:
            persona = Persona(**scene_input.persona)

        scene = Scene(
            id=scene_input.id,
            starting_prompt=scene_input.starting_prompt,
            conversation_plan=scene_input.conversation_plan,
            persona=persona,
            max_turns=scene_input.max_turns,
            expectations=Expectations(
                required_tools=scene_input.expectations.required_tools,
                forbidden_tools=scene_input.expectations.forbidden_tools,
            ),
        )

        backend = self._create_backend(model)
        simulator = UISimulator(
            backend=backend,
            conversation_plan=scene.conversation_plan,
            persona_prompt=scene.persona.to_prompt(),
        )

        trace = Trace(
            scene_id=scene.id,
            started_at=datetime.now(),
        )

        session = Session(
            id=session_id,
            scene=scene,
            simulator=simulator,
            trace=trace,
        )

        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def get_trace_dict(self, session: Session) -> dict[str, Any]:
        """Get the trace as a dictionary."""
        return {
            "scene_id": session.trace.scene_id,
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "arguments": tc.arguments,
                            "result": tc.result,
                        }
                        for tc in t.tool_calls
                    ],
                }
                for t in session.trace.turns
            ],
            "terminal_state": session.trace.terminal_state,
            "tool_calls": [
                {
                    "tool_name": tc.tool_name,
                    "arguments": tc.arguments,
                    "result": tc.result,
                }
                for tc in session.trace.tool_calls
            ],
        }
