"""Simulator: generates user turns from scene, persona, and conversation history."""

from typing import Protocol


class SimulatorBackend(Protocol):
    """Protocol for LLM backends that generate user turns."""

    def generate(self, prompt: str) -> str: ...


SIMULATOR_SYSTEM_PROMPT = """\
You are simulating a user in a customer interaction. You are NOT the agent.
You are the customer. Stay in character.

{persona}

CONVERSATION PLAN:
{conversation_plan}

RULES:
- Follow the conversation plan above. It tells you what you want and
  how to react to what the agent does.
- Respond naturally as a human would. Keep responses concise.
- If the plan says to provide information "if asked", wait until the
  agent asks before providing it.
- If you have accomplished your goal or the conversation has reached
  a natural end, respond with exactly: <finished>
- Never break character. Never mention that you are a simulator.
- Never use tool calls or function calls. You are the user, not the agent.
"""

FINISHED_SIGNAL = "<finished>"


class Simulator:
    """Generates synthetic user turns based on a scene and conversation history."""

    def __init__(
        self,
        backend: SimulatorBackend,
        conversation_plan: str,
        persona_prompt: str,
    ):
        self.backend = backend
        self.system_prompt = SIMULATOR_SYSTEM_PROMPT.format(
            persona=persona_prompt,
            conversation_plan=conversation_plan,
        )

    def next_turn(self, history: list[dict[str, str]]) -> str | None:
        """Generate the next user turn given conversation history.

        Returns None if the simulator signals the conversation is finished.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in history:
            messages.append(turn)

        # build a single prompt for simple backends
        prompt = self.system_prompt + "\n\nCONVERSATION SO FAR:\n"
        for turn in history:
            role = turn["role"].upper()
            prompt += f"{role}: {turn['content']}\n"
        prompt += "\nUSER:"

        response = self.backend.generate(prompt).strip()

        if FINISHED_SIGNAL in response.lower():
            return None
        return response
