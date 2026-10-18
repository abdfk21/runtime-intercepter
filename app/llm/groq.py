"""Groq LLM provider implementation using OpenAI-compatible REST API via httpx."""

import json
import os
from typing import Any, Dict, Optional, Type
import httpx
from pydantic import BaseModel
from app.core.logging import logger
from app.core.exceptions import LLMProviderError, LLMUnavailableError
from app.llm.base import BaseLLMProvider, ObservableReasoningArtifact


class GroqProvider(BaseLLMProvider):
    """Groq Cloud API LLM Provider."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        resolved_model = model_name or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
        super().__init__(resolved_model, temperature, max_tokens)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = "https://api.groq.com/openai/v1"

    def get_model_name(self) -> str:
        return self.model_name

    def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            with httpx.Client(timeout=5.0) as client:
                res = client.get(url, headers=headers)
                return res.status_code == 200
        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise LLMUnavailableError(
                "GROQ_API_KEY environment variable is not set. Please provide a key or use the Mock provider."
            )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    raise LLMProviderError(f"Groq API returned status {response.status_code}: {response.text}")
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.RequestError as e:
            raise LLMUnavailableError(f"Network error connecting to Groq API: {e}")
        except Exception as e:
            raise LLMProviderError(f"Groq generation error: {e}")

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel] = ObservableReasoningArtifact,
        system_prompt: Optional[str] = None,
    ) -> BaseModel:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        instructions = (
            f"{system_prompt or ''}\n\n"
            f"You MUST output valid JSON conforming exactly to the following JSON schema:\n"
            f"{schema_json}\n"
            f"Do not include markdown formatting or backticks outside the JSON string."
        )
        raw_text = self.generate(prompt=prompt, system_prompt=instructions)
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(cleaned)
            return schema.model_validate(parsed)
        except Exception as e:
            raise LLMProviderError(f"Failed to validate Groq structured output against schema: {e}. Output was: {raw_text}")
