"""Pydantic models for the HTTP simulator API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Affordance(BaseModel):
    """A UI element that the simulated user can interact with."""

    id: str
    type: Literal["text_input", "button", "link", "select", "checkbox", "radio"]
    selector: str
    label: str | None = None
    placeholder: str | None = None
    options: list[str] | None = None
    value: str | None = None
    checked: bool | None = None
    disabled: bool = False


class ActionTarget(BaseModel):
    """Target element for an action."""

    id: str | None = None
    selector: str


class Action(BaseModel):
    """An action the simulated user should take."""

    type: Literal["type", "click", "select", "check", "wait"]
    target: ActionTarget | None = None
    value: str | None = None
    checked: bool | None = None
    duration: int | None = None


class ExpectationsInput(BaseModel):
    """Expectations for validating the session."""

    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)


class SceneInput(BaseModel):
    """Scene configuration for starting a session."""

    id: str
    starting_prompt: str
    conversation_plan: str
    persona: str | dict[str, Any] = "cooperative"
    max_turns: int = 20
    expectations: ExpectationsInput = Field(default_factory=ExpectationsInput)


class CreateSessionRequest(BaseModel):
    """Request to create a new simulation session."""

    scene: SceneInput
    simulatorModel: str = "gpt-4o"


class CreateSessionResponse(BaseModel):
    """Response from creating a new session."""

    sessionId: str
    firstAction: Action


class ToolCallInput(BaseModel):
    """A tool call reported from the UI."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class TurnRequest(BaseModel):
    """Request to process a turn and get the next action."""

    displayedContent: str
    affordances: list[Affordance]
    toolCalls: list[ToolCallInput] = Field(default_factory=list)


class TurnResponse(BaseModel):
    """Response from processing a turn."""

    status: Literal["continue", "done"]
    action: Action | None = None
    reason: str | None = None


class CheckItem(BaseModel):
    """A single expectation check result."""

    label: str
    passed: bool
    detail: str


class EvaluateResponse(BaseModel):
    """Response from evaluating a session."""

    passed: bool
    checks: list[CheckItem]
    summary: str


class TraceResponse(BaseModel):
    """Response containing the session trace."""

    scene_id: str
    turns: list[dict[str, Any]]
    terminal_state: str | None = None
    tool_calls: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
