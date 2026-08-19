"""Core data models for understudy."""

import warnings
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field, ValidationError

from .validation import (
    SceneValidationError,
    check_common_mistakes,
    format_pydantic_error,
    validate_scene_data,
)


class PersonaPreset(StrEnum):
    """Built-in persona presets."""

    COOPERATIVE = "cooperative"
    FRUSTRATED_BUT_COOPERATIVE = "frustrated_but_cooperative"
    ADVERSARIAL = "adversarial"
    VAGUE = "vague"
    IMPATIENT = "impatient"


PERSONA_DESCRIPTIONS: dict[PersonaPreset, dict] = {
    PersonaPreset.COOPERATIVE: {
        "description": "Helpful and direct. Provides information when asked.",
        "behaviors": [
            "Answers questions directly and completely",
            "Provides requested information without hesitation",
            "Follows agent instructions cooperatively",
        ],
    },
    PersonaPreset.FRUSTRATED_BUT_COOPERATIVE: {
        "description": "Mildly frustrated but ultimately cooperative when asked "
        "clear questions.",
        "behaviors": [
            "Expresses mild frustration at the situation",
            "Pushes back once on denials before accepting",
            "Cooperates when the agent asks clear, direct questions",
            "May use short, clipped sentences",
        ],
    },
    PersonaPreset.ADVERSARIAL: {
        "description": "Tries to push boundaries and social-engineer exceptions.",
        "behaviors": [
            "Reframes requests to bypass policy",
            "Escalates language when denied",
            "Cites external authority (legal, regulatory)",
            "Does not accept the first denial",
            "May try to confuse or overwhelm the agent",
        ],
    },
    PersonaPreset.VAGUE: {
        "description": "Gives incomplete information, needs follow-up.",
        "behaviors": [
            "Provides partial answers to questions",
            "Omits details the agent needs",
            "Requires multiple follow-ups to get complete info",
            "May go off-topic occasionally",
        ],
    },
    PersonaPreset.IMPATIENT: {
        "description": "Wants fast resolution, dislikes long exchanges.",
        "behaviors": [
            "Gives very short answers",
            "Expresses impatience if the conversation drags",
            "Wants to get to resolution quickly",
            "May skip pleasantries",
        ],
    },
}


class Persona(BaseModel):
    """A user persona for the simulator to adopt."""

    description: str
    behaviors: list[str] = Field(default_factory=list)

    # The presets are attached below the class body, once from_preset exists to
    # build them. Declared here so they are part of the class rather than names
    # bolted on afterwards -- ClassVar keeps pydantic from reading them as
    # fields, which is what lets the assignment work at all.
    COOPERATIVE: ClassVar["Persona"]
    FRUSTRATED_BUT_COOPERATIVE: ClassVar["Persona"]
    ADVERSARIAL: ClassVar["Persona"]
    VAGUE: ClassVar["Persona"]
    IMPATIENT: ClassVar["Persona"]

    @classmethod
    def from_preset(cls, preset: PersonaPreset | str) -> "Persona":
        """Build a Persona from a preset enum value or its string name."""
        if isinstance(preset, str):
            preset = PersonaPreset(preset)
        data = PERSONA_DESCRIPTIONS[preset]
        return cls(**data)

    def to_prompt(self) -> str:
        """Render persona as a prompt fragment for the simulator."""
        lines = [f"User persona: {self.description}"]
        if self.behaviors:
            lines.append("Behaviors:")
            lines.extend(f"  - {b}" for b in self.behaviors)
        return "\n".join(lines)


# set presets as class attributes
Persona.COOPERATIVE = Persona.from_preset(PersonaPreset.COOPERATIVE)
Persona.FRUSTRATED_BUT_COOPERATIVE = Persona.from_preset(
    PersonaPreset.FRUSTRATED_BUT_COOPERATIVE
)
Persona.ADVERSARIAL = Persona.from_preset(PersonaPreset.ADVERSARIAL)
Persona.VAGUE = Persona.from_preset(PersonaPreset.VAGUE)
Persona.IMPATIENT = Persona.from_preset(PersonaPreset.IMPATIENT)


class Expectations(BaseModel):
    """What should and should not happen in a scene."""

    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    required_agents: list[str] = Field(default_factory=list)
    forbidden_agents: list[str] = Field(default_factory=list)
    required_agent_tools: dict[str, list[str]] = Field(default_factory=dict)
    expected_resolution: str | None = None
    metrics: list[str] = Field(default_factory=list)
    expected_trajectory: list[str] | None = None
    trajectory_match_mode: str = "exact"


class Scene(BaseModel):
    """A conversation fixture: the world, the user, and the expectations."""

    id: str
    description: str = ""

    # simulation
    starting_prompt: str
    conversation_plan: str
    persona: Persona
    max_turns: int = 20

    # world state
    context: dict[str, Any] = Field(default_factory=dict)

    # expectations
    expectations: Expectations = Field(default_factory=Expectations)

    @classmethod
    def from_file(cls, path: str | Path) -> "Scene":
        """Load a scene from a YAML or JSON file.

        Args:
            path: Path to the scene file (.yaml, .yml, or .json).

        Returns:
            The parsed Scene.

        Raises:
            SceneValidationError: If the scene file has validation errors or
                the YAML/JSON is malformed.
            FileNotFoundError: If the file doesn't exist.
        """
        path = Path(path)

        try:
            with path.open() as f:
                if path.suffix in (".yaml", ".yml"):
                    data = yaml.safe_load(f)
                else:
                    import json

                    data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Scene file not found: {path}") from e
        except yaml.YAMLError as e:
            raise SceneValidationError(
                f"Invalid YAML syntax in '{path}':\n  {e}", file_path=path
            ) from e

        if data is None:
            raise SceneValidationError(f"Scene file is empty: {path}", file_path=path)

        for mistake in check_common_mistakes(data):
            warnings.warn(f"{path}: {mistake}", UserWarning, stacklevel=2)

        validate_scene_data(data, file_path=path)

        try:
            return cls._from_dict(data)
        except ValidationError as e:
            raise SceneValidationError(
                format_pydantic_error(e, file_path=path), file_path=path
            ) from e

    @classmethod
    def _from_dict(cls, data: dict) -> "Scene":
        """Parse a scene dict, resolving persona presets."""
        persona_raw = data.get("persona")
        if isinstance(persona_raw, str):
            data["persona"] = Persona.from_preset(persona_raw)
        elif isinstance(persona_raw, dict):
            data["persona"] = Persona(**persona_raw)

        expectations_raw = data.get("expectations")
        if isinstance(expectations_raw, dict):
            data["expectations"] = Expectations(**expectations_raw)

        return cls(**data)
