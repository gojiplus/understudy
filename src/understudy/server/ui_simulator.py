"""UI Simulator: generates user actions based on displayed content and affordances."""

import json
from typing import Protocol

from .models import Action, ActionTarget, Affordance


class SimulatorBackend(Protocol):
    """Protocol for LLM backends that generate user actions."""

    def generate(self, prompt: str) -> str: ...


UI_SIMULATOR_SYSTEM_PROMPT = """\
You are simulating a user interacting with a web UI. You are NOT the agent/system.
You are the customer/user. Stay in character.

{persona}

CONVERSATION PLAN:
{conversation_plan}

RULES:
- Follow the conversation plan above. It tells you what you want and how to react.
- You will be shown the current UI state (what the agent/system displayed) and
  available affordances (buttons, inputs, etc.) you can interact with.
- Output a JSON action describing what UI action to take next.
- If you have accomplished your goal or the conversation has reached a natural end,
  output: {{"done": true, "reason": "finished"}}
- Never break character. Never mention that you are a simulator.
- Choose actions that make sense given the affordances available.
"""

ACTION_PROMPT_TEMPLATE = """\
CURRENT UI STATE:
The system/agent displayed:
{displayed_content}

AVAILABLE AFFORDANCES:
{affordances_json}

Based on your conversation plan and the current state, what action should you take?

Output one of:
1. Type: {{"type": "type", "target": {{"id": "x", "selector": "s"}}, "value": "text"}}
2. Click: {{"type": "click", "target": {{"id": "x", "selector": "s"}}}}
3. Select: {{"type": "select", "target": {{"id": "x", "selector": "s"}}, "value": "opt"}}
4. Check: {{"type": "check", "target": {{"id": "x", "selector": "s"}}, "checked": true}}
5. Wait: {{"type": "wait", "duration": 1000}}
6. Done: {{"done": true, "reason": "finished"}}

Output ONLY the JSON, no explanation.
YOUR ACTION:
"""


class UISimulator:
    """Generates synthetic user actions based on UI state and affordances."""

    def __init__(
        self,
        backend: SimulatorBackend,
        conversation_plan: str,
        persona_prompt: str,
    ):
        self.backend = backend
        self.system_prompt = UI_SIMULATOR_SYSTEM_PROMPT.format(
            persona=persona_prompt,
            conversation_plan=conversation_plan,
        )
        self.history: list[dict[str, str]] = []

    def get_first_action(self, starting_prompt: str, affordances: list[Affordance]) -> Action:
        """Generate the first action to type the starting prompt."""
        text_inputs = [a for a in affordances if a.type == "text_input" and not a.disabled]
        if text_inputs:
            target = text_inputs[0]
            return Action(
                type="type",
                target=ActionTarget(id=target.id, selector=target.selector),
                value=starting_prompt,
            )
        return Action(
            type="type",
            target=ActionTarget(id="chat-input", selector="input.chat-input"),
            value=starting_prompt,
        )

    def _strip_json_markers(self, text: str) -> str:
        """Strip markdown code fence markers from JSON response."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def next_action(
        self,
        displayed_content: str,
        affordances: list[Affordance],
    ) -> Action | None:
        """Generate the next action given the current UI state.

        Returns None if the simulator signals the conversation is finished.
        """
        self.history.append({"role": "assistant", "content": displayed_content})

        affordances_data = [a.model_dump(exclude_none=True) for a in affordances]
        affordances_json = json.dumps(affordances_data, indent=2)

        action_prompt = ACTION_PROMPT_TEMPLATE.format(
            displayed_content=displayed_content,
            affordances_json=affordances_json,
        )

        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in self.history
        )

        full_prompt = (
            self.system_prompt + "\n\nCONVERSATION SO FAR:\n" + history_text + "\n" + action_prompt
        )

        response = self._strip_json_markers(self.backend.generate(full_prompt))

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return Action(type="wait", duration=500)

        if data.get("done"):
            return None

        action_type = data.get("type")
        if action_type not in ("type", "click", "select", "check", "wait"):
            return Action(type="wait", duration=500)

        target = None
        if "target" in data and data["target"]:
            target = ActionTarget(
                id=data["target"].get("id"),
                selector=data["target"].get("selector", ""),
            )

        action = Action(
            type=action_type,
            target=target,
            value=data.get("value"),
            checked=data.get("checked"),
            duration=data.get("duration"),
        )

        if action_type == "type" and action.value:
            self.history.append({"role": "user", "content": action.value})

        return action
