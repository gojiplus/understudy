"""HTTP adapter: test deployed ADK agents via REST API."""

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from ..mocks import MockToolkit
from ..runner import AgentApp, AgentResponse
from ..trace import AgentTransfer, ToolCall


class HTTPApp(AgentApp):
    """Test deployed ADK agents via REST API.

    Usage:
        from understudy.http import HTTPApp

        app = HTTPApp(
            base_url="http://localhost:8080",
            app_name="customer_service",
        )
        trace = run(app, scene)
    """

    def __init__(
        self,
        base_url: str,
        app_name: str,
        user_id: str = "understudy_user",
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ):
        """Configure the HTTP client for a deployed agent.

        Args:
            base_url: Base URL of the deployed agent
                (e.g., "http://localhost:8080").
            app_name: Name of the ADK application.
            user_id: User ID for the session.
            headers: Additional HTTP headers (e.g., for authentication).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.app_name = app_name
        self.user_id = user_id
        self.headers = headers or {}
        self.timeout = timeout
        self._client = None
        self._session_id: str | None = None
        self._current_agent: str | None = None

    def start(self, mocks: MockToolkit | None = None) -> None:  # noqa: ARG002
        """Create an HTTP client and a remote session.

        Args:
            mocks: Accepted by the adapter protocol; remote tools are not replaced.

        Raises:
            ImportError: If httpx is not installed.
            RuntimeError: If the server rejects session creation.
        """
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx package required. Install with: pip install understudy[http]"
            ) from e

        self._client = httpx.Client(timeout=self.timeout, headers=self.headers)
        self._current_agent = None
        try:
            self._session_id = self._create_session()
        except Exception as e:
            self.stop()
            raise RuntimeError(f"Session creation failed: {e}") from e

    def _create_session(self) -> str:
        """Create the session required by ADK's run endpoint."""
        import uuid

        assert self._client is not None  # noqa: S101
        session_id = str(uuid.uuid4())
        url = (
            f"{self.base_url}/apps/{self.app_name}/users/{self.user_id}"
            f"/sessions/{session_id}"
        )
        response = self._client.post(url, json={})
        response.raise_for_status()
        return response.json()["id"]

    def send(self, message: str) -> AgentResponse:
        """Send a message and capture the streamed ADK events."""
        return self._send(
            f"{self.base_url}/run_sse",
            {
                "app_name": self.app_name,
                "user_id": self.user_id,
                "session_id": self._session_id,
                "new_message": {"role": "user", "parts": [{"text": message}]},
            },
        )

    def _send(self, url: str, payload: dict[str, Any]) -> AgentResponse:
        """Read one turn from the remote agent.

        Args:
            url: Streaming endpoint.
            payload: Request body for this transport.

        Returns:
            The text, tool events, transfers, and usage for this turn.

        Raises:
            RuntimeError: If the adapter is stopped or the request fails.
        """
        if self._client is None:
            raise RuntimeError("HTTPApp not started. Call start() first.")
        import httpx

        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                result = _parse_events(response.iter_lines(), self._current_agent)
        except httpx.HTTPError as e:
            raise RuntimeError(f"HTTP request failed: {e}") from e
        self._current_agent = result.agent_name
        return result

    def stop(self) -> None:
        """Close the HTTP client and forget local session state."""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._session_id = None
        self._current_agent = None


class AgentEngineApp(HTTPApp):
    """Test agents deployed on Google Cloud Agent Engine.

    Usage:
        from understudy.http import AgentEngineApp

        app = AgentEngineApp(
            project_id="my-project",
            location="us-central1",
            resource_id="my-agent-id",
        )
        trace = run(app, scene)
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        resource_id: str,
        app_name: str = "default",
        user_id: str = "understudy_user",
        credentials: Any = None,
        timeout: float = 60.0,
    ):
        """Configure the client for an Agent Engine deployment.

        Args:
            project_id: Google Cloud project ID.
            location: Agent Engine location (e.g., "us-central1").
            resource_id: Agent resource ID.
            app_name: Application name.
            user_id: User ID for the session.
            credentials: Optional Google credentials object used for auth.
                If None, uses default credentials.
            timeout: Request timeout in seconds.
        """
        base_url = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/{location}/"
            f"reasoningEngines/{resource_id}"
        )

        self._credentials = credentials
        self._project_id = project_id
        self._location = location
        self._resource_id = resource_id

        super().__init__(
            base_url=base_url,
            app_name=app_name,
            user_id=user_id,
            timeout=timeout,
        )

    def start(self, mocks=None) -> None:
        """Initialize HTTP client with Google Cloud authentication."""
        token = self._get_access_token()
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        super().start(mocks)

    def _get_access_token(self) -> str:
        """Get a valid access token from supplied or default credentials.

        Returns:
            The credentials' access token.

        Raises:
            ImportError: If google-auth is not installed.
            RuntimeError: If credentials cannot be loaded or refreshed.
        """
        try:
            import google.auth
            from google.auth.credentials import TokenState
            from google.auth.transport.requests import Request
        except ImportError as e:
            raise ImportError(
                "google-auth package required for Agent Engine. "
                "Install with: pip install google-auth"
            ) from e
        try:
            if self._credentials is None:
                self._credentials, _ = google.auth.default()
            if self._credentials.token_state != TokenState.FRESH:
                self._credentials.refresh(Request())
            return self._credentials.token
        except Exception as e:
            raise RuntimeError(f"Failed to get Google Cloud credentials: {e}") from e

    def _create_session(self) -> str:
        """Create a session through the Agent Engine query endpoint."""
        assert self._client is not None  # noqa: S101
        response = self._client.post(
            f"{self.base_url}:query",
            json={
                "class_method": "async_create_session",
                "input": {"user_id": self.user_id},
            },
        )
        response.raise_for_status()
        return response.json()["id"]

    def send(self, message: str) -> AgentResponse:
        """Send a message through Agent Engine's async_stream_query method."""
        return self._send(
            f"{self.base_url}:streamQuery",
            {
                "class_method": "async_stream_query",
                "input": {
                    "user_id": self.user_id,
                    "session_id": self._session_id,
                    "message": message,
                },
            },
        )


def _parse_events(lines: Iterable[str], current_agent: str | None) -> AgentResponse:
    """Decode ADK API server SSE or Agent Engine newline-delimited events."""
    calls: list[ToolCall] = []
    calls_by_id: dict[str, ToolCall] = {}
    transfers: list[AgentTransfer] = []
    text_parts: list[str] = []
    terminal_state = None
    input_tokens = output_tokens = thinking_tokens = 0

    for line in lines:
        data = line.removeprefix("data:").strip()
        if not data:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue

        actions = event.get("actions") or {}
        for target in (
            event.get("author"),
            actions.get("transferToAgent", actions.get("transfer_to_agent")),
        ):
            if target and target != current_agent:
                if current_agent:
                    transfers.append(
                        AgentTransfer(
                            from_agent=current_agent,
                            to_agent=target,
                            timestamp=datetime.now(UTC),
                        )
                    )
                current_agent = target
        if actions.get("escalate"):
            terminal_state = "agent_ended"

        content = event.get("content") or {}
        for part in content.get("parts") or []:
            if part.get("text") and not part.get("thought"):
                text_parts.append(part["text"])
            function_call = part.get("functionCall", part.get("function_call"))
            if function_call and function_call.get("name"):
                call = ToolCall(
                    tool_name=function_call["name"],
                    arguments=function_call.get("args") or {},
                    agent_name=current_agent,
                )
                calls.append(call)
                if function_call.get("id"):
                    calls_by_id[function_call["id"]] = call
            function_response = part.get(
                "functionResponse", part.get("function_response")
            )
            if function_response:
                if function_response.get("id"):
                    matched = calls_by_id.get(function_response["id"])
                else:
                    matched = next(
                        (
                            call
                            for call in calls
                            if call.tool_name == function_response.get("name")
                            and call.result is None
                        ),
                        None,
                    )
                if matched is not None:
                    matched.result = function_response.get("response")

        usage = event.get("usageMetadata", event.get("usage_metadata")) or {}
        input_tokens += (
            usage.get("promptTokenCount", usage.get("prompt_token_count")) or 0
        )
        output_tokens += (
            usage.get("candidatesTokenCount", usage.get("candidates_token_count")) or 0
        )
        thinking_tokens += (
            usage.get("thoughtsTokenCount", usage.get("thoughts_token_count")) or 0
        )

    return AgentResponse(
        content=" ".join(text_parts),
        tool_calls=calls,
        terminal_state=terminal_state,
        agent_name=current_agent,
        agent_transfers=transfers,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
    )
