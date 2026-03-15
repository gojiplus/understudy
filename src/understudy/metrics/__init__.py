"""Metrics module: pluggable evaluation metrics for traces."""

from .builtins import (
    compute_efficiency,
    compute_resolution_match,
    compute_tool_trajectory,
)
from .registry import MetricRegistry, MetricResult

__all__ = [
    "MetricRegistry",
    "MetricResult",
    "compute_efficiency",
    "compute_resolution_match",
    "compute_tool_trajectory",
]
