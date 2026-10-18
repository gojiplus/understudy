"""LangChain create_agent integration with pre-execution tool middleware."""

from importlib.metadata import version
from typing import Any

from ..langgraph import LangGraphApp


class LangChainApp(LangGraphApp):
    """Build a LangChain agent with isolated checkpoints and tool interception."""

    def __init__(
        self, model: Any, tools: list[Any], *, app_id: str = "langchain", **options: Any
    ):
        """Create an instrumented LangChain agent.

        Args:
            model: LangChain chat model or model identifier.
            tools: Tools passed to create_agent.
            app_id: Application revision identifier for checkpoints.
            **options: Additional create_agent options, including user middleware.
        """
        from langchain.agents import create_agent
        from langgraph.checkpoint.memory import InMemorySaver

        from .middleware import MockTools

        middleware = [MockTools(), *options.pop("middleware", [])]
        options.setdefault("checkpointer", InMemorySaver())
        graph = create_agent(model=model, tools=tools, middleware=middleware, **options)
        super().__init__(
            graph,
            app_id=f"{app_id}:langchain-{version('langchain')}",
            tool_interception=True,
        )
