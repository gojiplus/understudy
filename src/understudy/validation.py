"""Friendly validation error messages for understudy."""

from pathlib import Path
from typing import Any

from pydantic import ValidationError


class SceneValidationError(Exception):
    """Raised when a scene file fails validation with a friendly message."""

    def __init__(self, message: str, file_path: Path | None = None):
        self.message = message
        self.file_path = file_path
        super().__init__(message)


FIELD_EXAMPLES: dict[str, str] = {
    "starting_prompt": '"I need help with my order"',
    "conversation_plan": '"Goal: Return item from order ORD-123.\\nProvide order ID when asked."',
    "persona": '"cooperative"  # or: frustrated_but_cooperative, adversarial, vague, impatient',
    "id": '"my_scene_id"',
    "description": '"Customer wants to return an item"',
    "max_turns": "20",
    "context": "customer:\\n    name: Jane Doe",
    "expectations": "required_tools:\\n    - lookup_order",
}

FIELD_HELP: dict[str, str] = {
    "starting_prompt": "The first message the simulated user sends to start the conversation",
    "conversation_plan": "Instructions for the simulator about what the user wants to accomplish",
    "persona": "How the simulated user should behave (preset name or custom description)",
    "id": "Unique identifier for this scene",
    "max_turns": "Maximum number of conversation turns before stopping",
    "context": "World state data (customer info, orders, policies) available to mock tools",
    "expectations": "What should happen: required_tools, forbidden_tools, expected_resolution",
}


def format_pydantic_error(
    error: ValidationError,
    file_path: Path | None = None,
    data: dict[str, Any] | None = None,
) -> str:
    """Convert a Pydantic ValidationError into a friendly message."""
    lines = []

    if file_path:
        lines.append(f"Scene validation error in '{file_path}':")
    else:
        lines.append("Scene validation error:")

    lines.append("")

    for err in error.errors():
        loc = err.get("loc", ())
        field = ".".join(str(x) for x in loc) if loc else "root"
        err_type = err.get("type", "")
        msg = err.get("msg", "")

        if err_type == "missing":
            lines.append(f"  - Missing required field '{field}'")
            if field in FIELD_HELP:
                lines.append(f"    {FIELD_HELP[field]}")
        elif err_type == "string_type":
            lines.append(f"  - Field '{field}' must be a string")
        elif err_type == "int_type":
            lines.append(f"  - Field '{field}' must be an integer")
        elif err_type == "list_type":
            lines.append(f"  - Field '{field}' must be a list")
        elif err_type == "dict_type":
            lines.append(f"  - Field '{field}' must be a mapping/dict")
        elif "enum" in err_type.lower() or "literal" in err_type.lower():
            lines.append(f"  - Invalid value for '{field}': {msg}")
        else:
            lines.append(f"  - {field}: {msg}")

    example_fields = [
        ".".join(str(x) for x in err.get("loc", ()))
        for err in error.errors()
        if err.get("type") == "missing"
    ]
    examples_to_show = [f for f in example_fields if f in FIELD_EXAMPLES]

    if examples_to_show:
        lines.append("")
        lines.append("Example:")
        for field in examples_to_show[:3]:
            lines.append(f"  {field}: {FIELD_EXAMPLES[field]}")

    return "\n".join(lines)


def validate_scene_data(data: dict[str, Any], file_path: Path | None = None) -> None:
    """Validate scene data and raise friendly errors.

    Call this before creating a Scene to get better error messages.
    """
    warnings = []

    if "mocks" in data:
        warnings.append("Field 'mocks' is not used in scene files (mocks are provided at runtime)")

    context = data.get("context", {})
    expectations = data.get("expectations", {})

    if context and expectations:
        context_tools = set()
        if "orders" in context:
            context_tools.add("lookup_order")
        if "policy" in context:
            context_tools.add("get_return_policy")

        required = set(expectations.get("required_tools", []))
        for tool in required:
            if tool.startswith("create_") or tool.startswith("issue_"):
                continue
            if context_tools and tool not in context_tools:
                pass

    if warnings:
        import sys

        for w in warnings:
            print(f"Warning: {w}", file=sys.stderr)


def check_common_mistakes(data: dict[str, Any], file_path: Path | None = None) -> list[str]:
    """Check for common scene definition mistakes. Returns list of warnings."""
    warnings = []

    if "prompt" in data and "starting_prompt" not in data:
        warnings.append(
            "Found 'prompt' but expected 'starting_prompt'. Did you mean 'starting_prompt'?"
        )

    if "plan" in data and "conversation_plan" not in data:
        warnings.append(
            "Found 'plan' but expected 'conversation_plan'. Did you mean 'conversation_plan'?"
        )

    if "expected_tools" in data and "expectations" not in data:
        warnings.append(
            "Found 'expected_tools' at root level. "
            "Did you mean to put it under 'expectations.required_tools'?"
        )

    persona = data.get("persona")
    if isinstance(persona, str):
        valid_presets = {
            "cooperative",
            "frustrated_but_cooperative",
            "adversarial",
            "vague",
            "impatient",
        }
        if persona not in valid_presets:
            warnings.append(
                f"Unknown persona preset '{persona}'. "
                f"Valid presets: {', '.join(sorted(valid_presets))}. "
                f"Use a dict for custom personas: persona:\\n  description: '...'"
            )

    expectations = data.get("expectations", {})
    if isinstance(expectations, dict):
        trajectory_mode = expectations.get("trajectory_match_mode")
        if trajectory_mode and trajectory_mode not in ("exact", "subset", "ordered"):
            warnings.append(
                f"Unknown trajectory_match_mode '{trajectory_mode}'. "
                f"Valid modes: exact, subset, ordered"
            )

    return warnings
