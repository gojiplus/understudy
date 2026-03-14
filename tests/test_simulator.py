"""Tests for the Simulator class."""

from understudy import Simulator


class MockBackend:
    """Mock backend that returns predefined responses."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return "<finished>"


class TestSimulator:
    def test_next_turn_returns_response(self):
        backend = MockBackend(["I have a question about my order"])
        simulator = Simulator(
            backend=backend,
            conversation_plan="Ask about order status",
            persona_prompt="A friendly customer",
        )

        history = [
            {"role": "assistant", "content": "Hello, how can I help you?"},
        ]
        result = simulator.next_turn(history)

        assert result == "I have a question about my order"
        assert backend.call_count == 1

    def test_next_turn_includes_history_in_prompt(self):
        backend = MockBackend(["Follow-up question"])
        simulator = Simulator(
            backend=backend,
            conversation_plan="Ask about shipping",
            persona_prompt="Impatient customer",
        )

        history = [
            {"role": "user", "content": "Where is my package?"},
            {"role": "assistant", "content": "Let me check that for you."},
        ]
        simulator.next_turn(history)

        prompt = backend.prompts[0]
        assert "Where is my package?" in prompt
        assert "Let me check that for you" in prompt

    def test_next_turn_includes_persona(self):
        backend = MockBackend(["Response"])
        simulator = Simulator(
            backend=backend,
            conversation_plan="Test plan",
            persona_prompt="An elderly customer who is not tech-savvy",
        )

        simulator.next_turn([{"role": "assistant", "content": "Hi"}])

        prompt = backend.prompts[0]
        assert "elderly" in prompt
        assert "not tech-savvy" in prompt

    def test_next_turn_includes_conversation_plan(self):
        backend = MockBackend(["Response"])
        simulator = Simulator(
            backend=backend,
            conversation_plan="Step 1: Ask about refund. Step 2: Provide order number.",
            persona_prompt="Customer",
        )

        simulator.next_turn([{"role": "assistant", "content": "Hello"}])

        prompt = backend.prompts[0]
        assert "Ask about refund" in prompt
        assert "Provide order number" in prompt


class TestFinishedSignal:
    def test_finished_signal_returns_none(self):
        backend = MockBackend(["<finished>"])
        simulator = Simulator(
            backend=backend,
            conversation_plan="Complete the conversation",
            persona_prompt="Customer",
        )

        result = simulator.next_turn([{"role": "assistant", "content": "Goodbye!"}])

        assert result is None

    def test_finished_signal_case_insensitive(self):
        backend = MockBackend(["<FINISHED>"])
        simulator = Simulator(
            backend=backend,
            conversation_plan="Test",
            persona_prompt="Customer",
        )

        result = simulator.next_turn([{"role": "assistant", "content": "Done"}])

        assert result is None

    def test_finished_signal_with_surrounding_text(self):
        backend = MockBackend(["Thank you for your help! <finished>"])
        simulator = Simulator(
            backend=backend,
            conversation_plan="Test",
            persona_prompt="Customer",
        )

        result = simulator.next_turn([{"role": "assistant", "content": "Goodbye"}])

        assert result is None


class TestMultipleTurns:
    def test_multiple_turns_conversation(self):
        backend = MockBackend(
            [
                "What's my order status?",
                "The order number is ORD-12345",
                "When will it arrive?",
                "<finished>",
            ]
        )
        simulator = Simulator(
            backend=backend,
            conversation_plan="Ask about order, then shipping",
            persona_prompt="Customer",
        )

        history: list[dict[str, str]] = []
        results = []

        for agent_response in ["Hi!", "Let me check.", "It's on the way.", "Tomorrow."]:
            history.append({"role": "assistant", "content": agent_response})
            result = simulator.next_turn(history)
            if result is None:
                break
            results.append(result)
            history.append({"role": "user", "content": result})

        assert len(results) == 3
        assert "order" in results[0].lower()
        assert backend.call_count == 4
