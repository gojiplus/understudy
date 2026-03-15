"""HTTP simulator API for browser/UI testing."""

from fastapi import FastAPI, HTTPException

from ..check import check
from .models import (
    Action,
    ActionTarget,
    CheckItem,
    CreateSessionRequest,
    CreateSessionResponse,
    EvaluateResponse,
    TurnRequest,
    TurnResponse,
)
from .sessions import SessionManager

app = FastAPI(
    title="understudy HTTP Simulator",
    description="Simulation API for browser/UI testing with Cypress",
    version="0.1.0",
)

session_manager = SessionManager()


def get_app(model: str = "gpt-4o") -> FastAPI:
    """Get the FastAPI app with configured model."""
    session_manager.default_model = model
    return app


@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """Start a simulation session with a scene."""
    session = session_manager.create_session(
        scene_input=request.scene,
        model=request.simulatorModel,
    )

    first_action = Action(
        type="type",
        target=ActionTarget(id="chat-input", selector="input.chat-input"),
        value=session.scene.starting_prompt,
    )

    session.add_user_turn(session.scene.starting_prompt)

    return CreateSessionResponse(
        sessionId=session.id,
        firstAction=first_action,
    )


@app.post("/sessions/{session_id}/turn", response_model=TurnResponse)
def process_turn(session_id: str, request: TurnRequest) -> TurnResponse:
    """Report current UI state and get next user action."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.finished:
        return TurnResponse(
            status="done",
            reason=session.finish_reason or "finished",
        )

    if session.turn_count >= session.scene.max_turns:
        session.mark_finished("max_turns_reached")
        return TurnResponse(
            status="done",
            reason="max_turns_reached",
        )

    session.add_agent_turn(
        content=request.displayedContent,
        tool_calls=request.toolCalls if request.toolCalls else None,
    )

    action = session.simulator.next_action(
        displayed_content=request.displayedContent,
        affordances=request.affordances,
    )

    if action is None:
        session.mark_finished("finished")
        return TurnResponse(
            status="done",
            reason="finished",
        )

    if action.type == "type" and action.value:
        session.add_user_turn(action.value)

    return TurnResponse(
        status="continue",
        action=action,
    )


@app.post("/sessions/{session_id}/evaluate", response_model=EvaluateResponse)
def evaluate_session(session_id: str) -> EvaluateResponse:
    """Evaluate the conversation against scene expectations."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    check_result = check(session.trace, session.scene.expectations)

    checks = [
        CheckItem(label=c.label, passed=c.passed, detail=c.detail) for c in check_result.checks
    ]

    passed_count = sum(1 for c in checks if c.passed)
    total_count = len(checks)

    return EvaluateResponse(
        passed=check_result.passed,
        checks=checks,
        summary=f"{passed_count}/{total_count} checks passed",
    )


@app.get("/sessions/{session_id}/trace")
def get_trace(session_id: str) -> dict:
    """Get the raw trace for debugging."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session_manager.get_trace_dict(session)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """Cleanup session."""
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


__all__ = ["app", "get_app", "session_manager"]
