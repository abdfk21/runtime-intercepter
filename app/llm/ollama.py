"""Ollama local LLM provider implementation via REST API."""

import json
import os
from typing import Any, Dict, Optional, Type
import httpx
from pydantic import BaseModel
from app.core.logging import logger
from app.core.exceptions import LLMProviderError, LLMUnavailableError
from app.llm.base import BaseLLMProvider, ObservableReasoningArtifact


class OllamaProvider(BaseLLMProvider):
    """Local Ollama LLM Provider."""

    def __init__(
        self,
        model_name: str = "llama3",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        base_url: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens)
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")

    def get_model_name(self) -> str:
        return self.model_name

    def health_check(self) -> bool:
        try:
            url = f"{self.base_url}/api/tags"
            with httpx.Client(timeout=3.0) as client:
                res = client.get(url)
                return res.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                if response.status_code != 200:
                    raise LLMProviderError(f"Ollama returned status {response.status_code}: {response.text}")
                data = response.json()
                return data.get("response", "")
        except httpx.RequestError as e:
            raise LLMUnavailableError(
                f"Cannot connect to local Ollama server at {self.base_url}. Ensure Ollama is running or use the Mock provider: {e}"
            )
        except Exception as e:
            raise LLMProviderError(f"Ollama generation error: {e}")

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
            raise LLMProviderError(f"Failed to validate Ollama structured output against schema: {e}. Output was: {raw_text}")
