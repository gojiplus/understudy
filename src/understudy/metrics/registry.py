"""Metric registry: central registry for evaluation metrics."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import Expectations
    from ..trace import Trace


@dataclass
class MetricResult:
    """Result of computing a metric."""

    name: str
    passed: bool | None = None
    value: dict[str, Any] | None = None
    detail: str = ""

    def __repr__(self) -> str:
        if self.passed is not None:
            status = "passed" if self.passed else "failed"
            return f"MetricResult({self.name}: {status})"
        return f"MetricResult({self.name}: {self.value})"


@dataclass
class MetricDefinition:
    """Definition of a registered metric."""

    name: str
    compute_fn: Callable[["Trace", "Expectations"], MetricResult]
    description: str = ""


class MetricRegistry:
    """Central registry for evaluation metrics."""

    _metrics: dict[str, MetricDefinition] = field(default_factory=dict)
    _templates_dir: Path = Path(__file__).parent / "templates"

    @classmethod
    def register(cls, name: str, description: str = ""):
        """Decorator to register a metric computation function."""

        def decorator(fn: Callable[["Trace", "Expectations"], MetricResult]):
            cls._metrics[name] = MetricDefinition(
                name=name,
                compute_fn=fn,
                description=description,
            )
            return fn

        return decorator

    @classmethod
    def compute(cls, name: str, trace: "Trace", expectations: "Expectations") -> MetricResult:
        """Compute a metric by name."""
        if name not in cls._metrics:
            return MetricResult(
                name=name,
                passed=None,
                detail=f"Unknown metric: {name}",
            )
        return cls._metrics[name].compute_fn(trace, expectations)

    @classmethod
    def compute_all(
        cls, names: list[str], trace: "Trace", expectations: "Expectations"
    ) -> dict[str, MetricResult]:
        """Compute multiple metrics."""
        return {name: cls.compute(name, trace, expectations) for name in names}

    @classmethod
    def available_metrics(cls) -> list[str]:
        """List all registered metric names."""
        return list(cls._metrics.keys())

    @classmethod
    def get_template(cls, name: str) -> str | None:
        """Load a Jinja template for LLM-judged metrics."""
        template_path = cls._templates_dir / f"{name}.jinja"
        if template_path.exists():
            return template_path.read_text()
        return None


MetricRegistry._metrics = {}
