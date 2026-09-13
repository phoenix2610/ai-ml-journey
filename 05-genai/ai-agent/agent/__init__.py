"""An autonomous agent: plan, act, observe, with guardrails and traces."""

from agent.llm import GeminiClient, LLMError

__all__ = ["GeminiClient", "LLMError"]
__version__ = "0.1.0"
