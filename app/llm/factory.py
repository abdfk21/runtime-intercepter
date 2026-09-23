"""LLM Provider Factory with transparent fallback tracking."""

from typing import Any, Dict, Optional, Tuple
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import LLMUnavailableError
from app.llm.base import BaseLLMProvider
from app.llm.mock import MockProvider
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.ollama import OllamaProvider


class LLMFactory:
    """Creates LLM providers while tracking requested vs actual providers."""

    @staticmethod
    def create(
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        allow_fallback: bool = False,
        seed: int = 42,
    ) -> Tuple[BaseLLMProvider, Dict[str, Any]]:
        requested = (provider_name or settings.default_provider).lower()
        fallback_used = False
        actual = requested

        provider_map = {
            "mock": lambda: MockProvider(
                model_name=model_name or "deterministic-mock-v1", seed=seed
            ),
            "gemini": lambda: GeminiProvider(
                model_name=model_name or "gemini-2.5-flash"
            ),
            "groq": lambda: GroqProvider(
                model_name=model_name or "llama-3.3-70b-versatile"
            ),
            "ollama": lambda: OllamaProvider(
                model_name=model_name or "llama3"
            ),
        }

        if requested not in provider_map:
            raise ValueError(f"Unknown LLM provider requested: {requested}. Supported: {list(provider_map.keys())}")

        provider = provider_map[requested]()

        # Verify availability
        if not provider.health_check():
            if allow_fallback:
                logger.warning(
                    f"Requested provider '{requested}' is unavailable or lacks API keys. Falling back to MockProvider."
                )
                provider = provider_map["mock"]()
                actual = "mock"
                fallback_used = True
            else:
                raise LLMUnavailableError(
                    f"Requested provider '{requested}' is not available or has missing credentials. "
                    f"Set allow_fallback=True or configure the appropriate environment variable."
                )

        metadata = {
            "requested_provider": requested,
            "actual_provider": actual,
            "fallback_used": fallback_used,
            "model_name": provider.get_model_name(),
        }
        return provider, metadata
