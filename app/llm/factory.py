"""LLM Provider Factory with transparent fallback tracking."""

import os
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
                model_name=model_name or os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest"
            ),
            "groq": lambda: GroqProvider(
                model_name=model_name or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
            ),
            "ollama": lambda: OllamaProvider(
                model_name=model_name or os.getenv("OLLAMA_MODEL") or "llama3"
            ),
        }

        if requested not in provider_map:
            raise ValueError(f"Unknown LLM provider requested: {requested}. Supported: {list(provider_map.keys())}")

        provider = provider_map[requested]()
        fallback_reason = None

        # Verify availability
        if not provider.health_check():
            if allow_fallback:
                fallback_reason = f"Provider '{requested}' failed health check or lacks credentials. Fallback to mock."
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
            "fallback_reason": fallback_reason,
            "model_name": provider.get_model_name(),
        }
        return provider, metadata
