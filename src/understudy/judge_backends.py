"""Pluggable backends for LLM judge evaluations."""

import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class JudgeBackend(Protocol):
    """Protocol for judge backends.

    Implement this protocol to create custom judge backends
    for different LLM providers or evaluation methods.
    """

    def evaluate(self, prompt: str) -> str:
        """Evaluate a prompt and return the response.

        Args:
            prompt: The full evaluation prompt including conversation and rubric.

        Returns:
            The model's response (typically "YES" or "NO").
        """
        ...

    async def evaluate_async(self, prompt: str) -> str:
        """Asynchronous version of evaluate.

        Default implementation wraps the sync method.
        Override for true async support.
        """
        ...


class BaseJudgeBackend(ABC):
    """Base class for judge backends with common functionality."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 10,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def evaluate(self, prompt: str) -> str:
        """Evaluate a prompt and return the response."""
        pass

    async def evaluate_async(self, prompt: str) -> str:
        """Async evaluation - default wraps sync method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, prompt)


class LiteLLMBackend(BaseJudgeBackend):
    """Default backend using LiteLLM for unified provider access.

    Supports all providers that LiteLLM supports (OpenAI, Anthropic, etc.)
    via the unified interface.

    Example:
        backend = LiteLLMBackend(model="gpt-4o")
        backend = LiteLLMBackend(model="claude-sonnet-4-20250514")
        backend = LiteLLMBackend(model="gemini/gemini-pro")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 1.0,
        max_tokens: int = 10,
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self._litellm = None

    def _get_litellm(self):
        if self._litellm is None:
            try:
                import litellm

                self._litellm = litellm
            except ImportError as e:
                raise ImportError(
                    "litellm package required for LiteLLMBackend. Install with: pip install litellm"
                ) from e
        return self._litellm

    def evaluate(self, prompt: str) -> str:
        """Evaluate using LiteLLM."""
        litellm = self._get_litellm()
        response = litellm.completion(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    async def evaluate_async(self, prompt: str) -> str:
        """Async evaluation using LiteLLM's async support."""
        litellm = self._get_litellm()
        response = await litellm.acompletion(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class CallbackBackend(BaseJudgeBackend):
    """Backend that uses a custom callback function.

    Useful for testing or integrating with custom LLM wrappers.

    Example:
        def my_evaluator(prompt: str) -> str:
            return "YES" if "good" in prompt else "NO"

        backend = CallbackBackend(my_evaluator)
    """

    def __init__(
        self,
        callback,
        async_callback=None,
    ):
        super().__init__()
        self._callback = callback
        self._async_callback = async_callback

    def evaluate(self, prompt: str) -> str:
        """Evaluate using the callback function."""
        return self._callback(prompt)

    async def evaluate_async(self, prompt: str) -> str:
        """Async evaluation using async callback if provided."""
        if self._async_callback:
            return await self._async_callback(prompt)
        return await super().evaluate_async(prompt)
