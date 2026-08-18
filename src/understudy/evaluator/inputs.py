"""Input processors for the evaluator agent.

Parses agent code, docs, natural language specs, and golden traces
to extract capabilities, constraints, and expected behaviors.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CapabilityType(StrEnum):
    """Types of agent capabilities."""

    TOOL_USE = "tool_use"
    INFORMATION_RETRIEVAL = "information_retrieval"
    DECISION_MAKING = "decision_making"
    MULTI_STEP_TASK = "multi_step_task"
    ERROR_HANDLING = "error_handling"
    USER_INTERACTION = "user_interaction"
    STATE_MANAGEMENT = "state_management"
    POLICY_ENFORCEMENT = "policy_enforcement"
    OTHER = "other"


class ConstraintType(StrEnum):
    """Types of agent constraints."""

    MUST_DO = "must_do"
    MUST_NOT_DO = "must_not_do"
    CONDITIONAL = "conditional"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"


class Capability(BaseModel):
    """A single capability of the agent under test."""

    id: str
    name: str
    description: str
    capability_type: CapabilityType = CapabilityType.OTHER
    tools_involved: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    edge_cases: list[str] = Field(default_factory=list)
    priority: int = 1  # 1 = highest priority


class Constraint(BaseModel):
    """A constraint or policy the agent must follow."""

    id: str
    name: str
    description: str
    constraint_type: ConstraintType
    applies_when: str | None = None
    examples: list[str] = Field(default_factory=list)
    severity: str = "error"  # error, warning, info


class AgentSpec(BaseModel):
    """Structured representation of agent capabilities and constraints.

    This is the output of the input processor and serves as the basis
    for test generation.
    """

    name: str
    description: str = ""
    version: str | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    known_failure_modes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def capability_by_id(self, capability_id: str) -> Capability | None:
        for cap in self.capabilities:
            if cap.id == capability_id:
                return cap
        return None

    def capabilities_by_type(self, cap_type: CapabilityType) -> list[Capability]:
        return [c for c in self.capabilities if c.capability_type == cap_type]

    def capabilities_by_tool(self, tool_name: str) -> list[Capability]:
        return [c for c in self.capabilities if tool_name in c.tools_involved]


@dataclass
class InputSource:
    """A source of information about the agent."""

    source_type: str  # code, docs, spec, trace, scene
    path: Path | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InputProcessor:
    """Processes various input sources to extract an AgentSpec.

    Uses LLM to understand code, docs, and specs and extract
    structured information about agent capabilities.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.model = model

    def process(
        self,
        code: str | Path | None = None,
        docs: str | Path | None = None,
        spec: str | None = None,
        golden_traces: list[Any] | None = None,
        existing_scenes: list[Any] | None = None,
    ) -> AgentSpec:
        """Process all input sources and return a unified AgentSpec.

        Args:
            code: Path to agent code directory or code string
            docs: Path to documentation directory or docs string
            spec: Natural language specification
            golden_traces: List of known-good execution traces
            existing_scenes: Existing Scene objects to learn from

        Returns:
            AgentSpec with extracted capabilities and constraints
        """
        sources: list[InputSource] = []

        if code:
            sources.append(self._prepare_code_source(code))
        if docs:
            sources.append(self._prepare_docs_source(docs))
        if spec:
            sources.append(InputSource(source_type="spec", content=spec))
        if golden_traces:
            for i, trace in enumerate(golden_traces):
                sources.append(self._prepare_trace_source(trace, i))
        if existing_scenes:
            for i, scene in enumerate(existing_scenes):
                sources.append(self._prepare_scene_source(scene, i))

        if not sources:
            raise ValueError("At least one input source must be provided")

        return self._extract_spec(sources)

    def _prepare_code_source(self, code: str | Path) -> InputSource:
        if isinstance(code, Path) or (isinstance(code, str) and Path(code).exists()):
            path = Path(code)
            content = self._read_code_directory(path)
            return InputSource(source_type="code", path=path, content=content)
        return InputSource(source_type="code", content=code)

    def _prepare_docs_source(self, docs: str | Path) -> InputSource:
        if isinstance(docs, Path) or (isinstance(docs, str) and Path(docs).exists()):
            path = Path(docs)
            content = self._read_docs_directory(path)
            return InputSource(source_type="docs", path=path, content=content)
        return InputSource(source_type="docs", content=docs)

    def _prepare_trace_source(self, trace: Any, index: int) -> InputSource:
        if hasattr(trace, "model_dump_json"):
            content = trace.model_dump_json(indent=2)
        elif hasattr(trace, "conversation_text"):
            content = trace.conversation_text()
        else:
            content = str(trace)
        return InputSource(
            source_type="trace",
            content=content,
            metadata={"index": index},
        )

    def _prepare_scene_source(self, scene: Any, index: int) -> InputSource:
        if hasattr(scene, "model_dump_json"):
            content = scene.model_dump_json(indent=2)
        else:
            content = str(scene)
        return InputSource(
            source_type="scene",
            content=content,
            metadata={"index": index},
        )

    def _read_code_directory(self, path: Path) -> str:
        lines = []
        for py_file in sorted(path.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            rel_path = py_file.relative_to(path)
            lines.append(f"# File: {rel_path}")
            lines.append(py_file.read_text())
            lines.append("")
        return "\n".join(lines)

    def _read_docs_directory(self, path: Path) -> str:
        lines = []
        for doc_file in sorted(path.rglob("*")):
            if doc_file.is_file() and doc_file.suffix in (".md", ".rst", ".txt"):
                rel_path = doc_file.relative_to(path)
                lines.append(f"# File: {rel_path}")
                lines.append(doc_file.read_text())
                lines.append("")
        return "\n".join(lines)

    def _extract_spec(self, sources: list[InputSource]) -> AgentSpec:
        """Use LLM to extract structured spec from sources."""
        import litellm

        prompt = self._build_extraction_prompt(sources)

        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0,
        )

        content = response.choices[0].message.content
        if not content:
            return AgentSpec(name="unknown")

        import json

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return AgentSpec(name="unknown", description=content[:500])

        capabilities = []
        for i, cap_data in enumerate(data.get("capabilities", [])):
            cap = Capability(
                id=cap_data.get("id", f"cap_{i}"),
                name=cap_data.get("name", f"Capability {i}"),
                description=cap_data.get("description", ""),
                capability_type=CapabilityType(cap_data.get("type", "other")),
                tools_involved=cap_data.get("tools_involved", []),
                preconditions=cap_data.get("preconditions", []),
                postconditions=cap_data.get("postconditions", []),
                edge_cases=cap_data.get("edge_cases", []),
                priority=cap_data.get("priority", 1),
            )
            capabilities.append(cap)

        constraints = []
        for i, con_data in enumerate(data.get("constraints", [])):
            con = Constraint(
                id=con_data.get("id", f"con_{i}"),
                name=con_data.get("name", f"Constraint {i}"),
                description=con_data.get("description", ""),
                constraint_type=ConstraintType(con_data.get("type", "must_do")),
                applies_when=con_data.get("applies_when"),
                examples=con_data.get("examples", []),
                severity=con_data.get("severity", "error"),
            )
            constraints.append(con)

        return AgentSpec(
            name=data.get("name", "unknown"),
            description=data.get("description", ""),
            version=data.get("version"),
            capabilities=capabilities,
            constraints=constraints,
            available_tools=data.get("available_tools", []),
            known_failure_modes=data.get("known_failure_modes", []),
        )

    def _build_extraction_prompt(self, sources: list[InputSource]) -> str:
        source_texts = []
        for source in sources:
            if source.content:
                truncated = source.content[:10000]
                source_texts.append(
                    f"=== {source.source_type.upper()} ===\n{truncated}"
                )

        return f"""Analyze the following information about an AI agent and extract its capabilities and constraints.

{chr(10).join(source_texts)}

Return a JSON object with this structure:
{{
    "name": "agent name",
    "description": "brief description of what the agent does",
    "version": "version if known, or null",
    "capabilities": [
        {{
            "id": "unique_id",
            "name": "capability name",
            "description": "what this capability does",
            "type": "tool_use|information_retrieval|decision_making|multi_step_task|error_handling|user_interaction|state_management|policy_enforcement|other",
            "tools_involved": ["tool1", "tool2"],
            "preconditions": ["condition that must be true"],
            "postconditions": ["result after capability is used"],
            "edge_cases": ["edge case to test"],
            "priority": 1
        }}
    ],
    "constraints": [
        {{
            "id": "unique_id",
            "name": "constraint name",
            "description": "what the agent must/must not do",
            "type": "must_do|must_not_do|conditional|efficiency|safety",
            "applies_when": "condition when constraint applies",
            "examples": ["example of constraint"],
            "severity": "error|warning|info"
        }}
    ],
    "available_tools": ["tool1", "tool2"],
    "known_failure_modes": ["failure mode 1", "failure mode 2"]
}}

Be thorough in identifying capabilities and constraints. Consider:
- What tools can the agent use and when?
- What policies must the agent follow?
- What are the edge cases and failure modes?
- What are the expected inputs and outputs?"""
