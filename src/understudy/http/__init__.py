"""HTTP adapter: test deployed ADK agents via REST API."""

import json
from datetime import UTC, datetime
from typing import Any

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
        """
        Args:
            base_url: Base URL of the deployed agent (e.g., "http://localhost:8080").
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
        self._agent_transfers: list[AgentTransfer] = []

    def start(self, mocks=None) -> None:
        """Initialize HTTP client and create a session."""
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx package required. Install with: pip install understudy[http]"
            ) from e

        self._client = httpx.Client(timeout=self.timeout, headers=self.headers)
        self._current_agent = None
        self._agent_transfers = []

        import uuid

        self._session_id = str(uuid.uuid4())

    def send(self, message: str) -> AgentResponse:
        """Send a message to the deployed agent and capture the response."""
        if not self._client:
            raise RuntimeError("HTTPApp not started. Call start() first.")

        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "httpx package required. Install with: pip install understudy[http]"
            ) from e

        url = f"{self.base_url}/run_sse"
        payload = {
            "app_name": self.app_name,
            "user_id": self.user_id,
            "session_id": self._session_id,
            "new_message": {
                "role": "user",
                "parts": [{"text": message}],
            },
        }

        tool_calls: list[ToolCall] = []
        agent_text_parts: list[str] = []
        terminal_state: str | None = None
        current_agent = self._current_agent

        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data:
                        continue

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_agent = event.get("author")
                    if event_agent and event_agent != current_agent:
                        if current_agent:
                            self._agent_transfers.append(
                                AgentTransfer(
                                    from_agent=current_agent,
                                    to_agent=event_agent,
                                    timestamp=datetime.now(UTC),
                                )
                            )
                        current_agent = event_agent

                    actions = event.get("actions", {})
                    if actions and actions.get("transfer_to_agent"):
                        target = actions["transfer_to_agent"]
                        if current_agent and target != current_agent:
                            self._agent_transfers.append(
                                AgentTransfer(
                                    from_agent=current_agent,
                                    to_agent=target,
                                    timestamp=datetime.now(UTC),
                                )
                            )
                        current_agent = target

                    for fc in event.get("function_calls", []):
                        call = ToolCall(
                            tool_name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                            agent_name=current_agent,
                        )
                        tool_calls.append(call)

                    content = event.get("content", {})
                    for part in content.get("parts", []):
                        text = part.get("text")
                        if text:
                            agent_text_parts.append(text)
                            if "TERMINAL_STATE:" in text:
                                state = text.split("TERMINAL_STATE:")[-1].strip()
                                terminal_state = state.split()[0].strip()

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP request failed: {e.response.status_code}") from e

        self._current_agent = current_agent

        response = AgentResponse(
            content=" ".join(agent_text_parts),
            tool_calls=tool_calls,
            terminal_state=terminal_state,
            agent_name=current_agent,
            agent_transfers=list(self._agent_transfers),
        )
        return response

    def stop(self) -> None:
        """Clean up HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
        self._session_id = None
        self._agent_transfers = []


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
        """
        Args:
            project_id: Google Cloud project ID.
            location: Agent Engine location (e.g., "us-central1").
            resource_id: Agent resource ID.
            app_name: Application name.
            user_id: User ID for the session.
            credentials: Google Cloud credentials. If None, uses default credentials.
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
        """Get access token from credentials."""
        if self._credentials:
            if hasattr(self._credentials, "token"):
                return self._credentials.token
            if hasattr(self._credentials, "refresh"):
                from google.auth.transport.requests import Request

                self._credentials.refresh(Request())
                return self._credentials.token

        try:
            import google.auth
            import google.auth.transport.requests

            creds, _ = google.auth.default()
            creds.refresh(google.auth.transport.requests.Request())
            return creds.token
        except ImportError as e:
            raise ImportError(
                "google-auth package required for Agent Engine. "
                "Install with: pip install google-auth"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to get Google Cloud credentials: {e}") from e

    def send(self, message: str) -> AgentResponse:
        """Send a message to Agent Engine."""
        if not self._client:
            raise RuntimeError("AgentEngineApp not started. Call start() first.")

        url = f"{self.base_url}:streamQuery"
        payload = {
            "input": {
                "app_name": self.app_name,
                "user_id": self.user_id,
                "session_id": self._session_id,
                "new_message": {
                    "role": "user",
                    "parts": [{"text": message}],
                },
            },
        }

        tool_calls: list[ToolCall] = []
        agent_text_parts: list[str] = []
        terminal_state: str | None = None
        current_agent = self._current_agent

        try:
            with self._client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_agent = event.get("author")
                    if event_agent and event_agent != current_agent:
                        if current_agent:
                            self._agent_transfers.append(
                                AgentTransfer(
                                    from_agent=current_agent,
                                    to_agent=event_agent,
                                    timestamp=datetime.now(UTC),
                                )
                            )
                        current_agent = event_agent

                    actions = event.get("actions", {})
                    if actions and actions.get("transfer_to_agent"):
                        target = actions["transfer_to_agent"]
                        if current_agent and target != current_agent:
                            self._agent_transfers.append(
                                AgentTransfer(
                                    from_agent=current_agent,
                                    to_agent=target,
                                    timestamp=datetime.now(UTC),
                                )
                            )
                        current_agent = target

                    for fc in event.get("function_calls", []):
                        call = ToolCall(
                            tool_name=fc.get("name", ""),
                            arguments=fc.get("args", {}),
                            agent_name=current_agent,
                        )
                        tool_calls.append(call)

                    content = event.get("content", {})
                    for part in content.get("parts", []):
                        text = part.get("text")
                        if text:
                            agent_text_parts.append(text)
                            if "TERMINAL_STATE:" in text:
                                state = text.split("TERMINAL_STATE:")[-1].strip()
                                terminal_state = state.split()[0].strip()

        except Exception as e:
            raise RuntimeError(f"Agent Engine request failed: {e}") from e

        self._current_agent = current_agent

        return AgentResponse(
            content=" ".join(agent_text_parts),
            tool_calls=tool_calls,
            terminal_state=terminal_state,
            agent_name=current_agent,
            agent_transfers=list(self._agent_transfers),
        )
