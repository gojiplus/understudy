"""Invocation identities derived from native runtime paths."""

from .trace import AgentExecution


class ExecutionRecorder:
    """Record distinct invocations and their native parent paths."""

    def __init__(self) -> None:
        """Create an independent recorder for one adapter session."""
        self.executions: list[AgentExecution] = []
        self.by_id: dict[str, AgentExecution] = {}

    def observe(self, path: str, invocation: str) -> AgentExecution:
        """Register a native path and its ancestors without merging repeated calls."""
        segments = path.split("/")
        parent = None
        names = []
        for index, segment in enumerate(segments, 1):
            names.append(segment.rsplit("@", 1)[0])
            key = f"{invocation}:{'/'.join(segments[:index])}"
            if key not in self.by_id:
                execution = AgentExecution(
                    id=key,
                    agent_name=names[-1],
                    agent_path=tuple(names),
                    parent_id=parent,
                )
                self.by_id[key] = execution
                self.executions.append(execution)
            parent = key
        return self.by_id[f"{invocation}:{path}"]

    def finish(self, error: str | None = None) -> None:
        """Close any invocations still open when the enclosing run finishes."""
        for execution in self.executions:
            if execution.status == "running":
                execution.status = "error" if error else "completed"
                execution.error = error
