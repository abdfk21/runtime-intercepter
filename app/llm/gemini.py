"""Google Gemini provider implementation using REST API via httpx."""

import json
import os
from typing import Any, Dict, Optional, Type
import httpx
from pydantic import BaseModel
from app.core.logging import logger
from app.core.exceptions import LLMProviderError, LLMUnavailableError
from app.llm.base import BaseLLMProvider, ObservableReasoningArtifact


class GeminiProvider(BaseLLMProvider):
    """Google Gemini AI Studio LLM Provider."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ):
        resolved_model = model_name or os.getenv("GEMINI_MODEL") or "gemini-flash-lite-latest"
        super().__init__(resolved_model, temperature, max_tokens)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def get_model_name(self) -> str:
        return self.model_name

    def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            with httpx.Client(timeout=5.0) as client:
                res = client.get(url)
                return res.status_code == 200
        except Exception as e:
            logger.warning(f"Gemini health check failed: {e}")
            return False

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise LLMUnavailableError(
                "GEMINI_API_KEY environment variable is not set. Please provide a key or use the Mock provider."
            )

        candidate_models = [self.model_name]
        fallback_model = "gemini-flash-lite-latest"
        if self.model_name != fallback_model:
            candidate_models.append(fallback_model)

        import time
        max_retries = 3
        last_error = None

        for model_to_try in candidate_models:
            url = f"{self.base_url}/models/{model_to_try}:generateContent?key={self.api_key}"
            contents = []
            if system_prompt:
                contents.append({"role": "user", "parts": [{"text": f"System Directive: {system_prompt}"}]})
                contents.append({"role": "model", "parts": [{"text": "Understood."}]})
            contents.append({"role": "user", "parts": [{"text": prompt}]})

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            }

            for attempt in range(max_retries):
                try:
                    with httpx.Client(timeout=30.0) as client:
                        response = client.post(url, json=payload)
                        if response.status_code == 429 and "RESOURCE_EXHAUSTED" in response.text and model_to_try != fallback_model:
                            logger.warning(
                                f"Gemini quota exhausted for '{model_to_try}'. Falling back to active model '{fallback_model}'."
                            )
                            self.model_name = fallback_model
                            break  # Break inner retry loop to try fallback_model

                        if response.status_code in [503, 429] and attempt < max_retries - 1:
                            logger.warning(f"Gemini {response.status_code} received (attempt {attempt+1}/{max_retries}). Retrying in 2s...")
                            time.sleep(2.0)
                            continue

                        if response.status_code == 404 and model_to_try != fallback_model:
                            logger.warning(f"Gemini model '{model_to_try}' returned 404. Falling back to '{fallback_model}'.")
                            self.model_name = fallback_model
                            break

                        if response.status_code != 200:
                            raise LLMProviderError(f"Gemini API returned status {response.status_code}: {response.text}")

                        data = response.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise LLMProviderError(f"No candidates returned by Gemini: {data}")
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for p in reversed(parts):
                            if "text" in p and p["text"]:
                                self.model_name = model_to_try
                                return p["text"]
                        raise LLMProviderError(f"No text part found in Gemini candidate: {candidates[0]}")
                except httpx.RequestError as e:
                    last_error = LLMUnavailableError(f"Network error connecting to Gemini API: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2.0)
                        continue
                    raise last_error
                except LLMProviderError as e:
                    last_error = e
                    if attempt == max_retries - 1 and model_to_try == candidate_models[-1]:
                        raise e
                except Exception as e:
                    last_error = LLMProviderError(f"Gemini generation error: {e}")
                    if attempt == max_retries - 1 and model_to_try == candidate_models[-1]:
                        raise last_error

        if last_error:
            raise last_error

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
        # Extract JSON substring
        cleaned = raw_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(cleaned)
            return schema.model_validate(parsed)
        except Exception as e:
            raise LLMProviderError(f"Failed to validate Gemini structured output against schema: {e}. Output was: {raw_text}")
