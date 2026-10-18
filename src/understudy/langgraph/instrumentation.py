"""Record graph invocations using public LangChain callback events."""

from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler

from ..trace import AgentExecution


class GraphRecorder(BaseCallbackHandler):
    """Keep invocation identity across nested and parallel graph nodes."""

    def __init__(self):
        """Initialize a recorder for one turn."""
        self.executions: dict[str, AgentExecution] = {}
        self.parents: dict[str, str | None] = {}
        self.lock = Lock()

    def on_chain_start(
        self,
        serialized,
        inputs,  # noqa: ARG002
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        **kwargs,
    ):
        """Record visible graph nodes, retaining native run IDs."""
        with self.lock:
            key, parent = str(run_id), str(parent_run_id) if parent_run_id else None
            self.parents[key] = parent
            if "langsmith:hidden" in (tags or []):
                return
            while parent and parent not in self.executions:
                parent = self.parents.get(parent)
            name = kwargs.get("name") or (serialized or {}).get("name") or "graph"
            ancestor = self.executions.get(parent) if parent else None
            self.executions[key] = AgentExecution(
                id=key,
                parent_id=parent,
                agent_name=name,
                agent_path=(*ancestor.agent_path, name) if ancestor else (name,),
            )

    def on_chain_end(self, outputs, *, run_id, **kwargs):  # noqa: ARG002
        """Close successful invocations."""
        with self.lock:
            execution = self.executions.get(str(run_id))
            if execution:
                execution.status = "completed"

    def on_chain_error(self, error, *, run_id, **kwargs):  # noqa: ARG002
        """Close failed invocations."""
        with self.lock:
            execution = self.executions.get(str(run_id))
            if execution:
                execution.status, execution.error = "error", str(error)
