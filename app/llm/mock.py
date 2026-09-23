"""Deterministic offline mock LLM provider for zero-cost reproduction and tests."""

import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from app.llm.base import BaseLLMProvider, ObservableClaim, ObservableReasoningArtifact


class MockProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider that generates structured research artifacts offline."""

    def __init__(
        self,
        model_name: str = "deterministic-mock-v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
    ):
        super().__init__(model_name, temperature, max_tokens)
        self.seed = seed
        self.rng = random.Random(seed)

    def get_model_name(self) -> str:
        return self.model_name

    def health_check(self) -> bool:
        return True

    def _derive_seed_from_text(self, text: str) -> int:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        local_seed = self._derive_seed_from_text(prompt)
        rng = random.Random(local_seed)
        words = [
            "analysis", "confirmed", "evidence", "findings", "systematic",
            "cross-referenced", "parameters", "consistent", "verified", "data"
        ]
        return f"Mock response for [{prompt[:40]}...]: { ' '.join(rng.sample(words, 6)) }."

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel] = ObservableReasoningArtifact,
        system_prompt: Optional[str] = None,
    ) -> BaseModel:
        local_seed = self._derive_seed_from_text(prompt + (system_prompt or ""))
        rng = random.Random(local_seed)

        # Domain-aware deterministic response synthesis based on prompt content
        prompt_lower = prompt.lower()
        has_repair = "critical correction request" in prompt_lower
        has_warning = "safety warning" in prompt_lower or "exercise skepticism" in prompt_lower

        # Downstream cascade propagation: if upstream agent passed corrupted context and no repair or warning occurred
        corrupted_tokens = [
            "sydney", "1492", "9942", "austria", "tachyon", "inverted thermodynamics",
            "mutated finding", "unexplained units", "contrary to consensus", "adopting upstream finding"
        ]
        detected_corruptions = [tok for tok in corrupted_tokens if tok in prompt_lower]

        if not has_repair and not has_warning and detected_corruptions:
            corrupt_tok = detected_corruptions[0]
            answer = f"Analytical verification adopts upstream premise: {corrupt_tok} is confirmed across multi-agent findings."
            claims = [
                ObservableClaim(
                    claim=f"Assertion adopting upstream finding: {corrupt_tok} accepted into synthesized model.",
                    confidence=0.91,
                    evidence=[f"Propagated directly from preceding agent handoff containing {corrupt_tok}."]
                ),
                ObservableClaim(
                    claim=f"Corroborating correlation regarding {corrupt_tok} based on uninspected upstream context.",
                    confidence=0.87,
                    evidence=["Cross-agent handoff stream."]
                )
            ]
            assumptions = ["Upstream handoff assertions assumed valid without independent verification."]
            sources = ["Upstream Agent Handoff Stream"]
        elif "australia" in prompt_lower and "capital" in prompt_lower:
            answer = "Canberra is the federal capital of Australia, established following the 1913 foundation."
            claims = [
                ObservableClaim(
                    claim="Canberra is the capital city of Australia.",
                    confidence=0.98,
                    evidence=["Australian Constitution 1901 Section 125 designates federal territory."]
                ),
                ObservableClaim(
                    claim="Sydney is the largest city by population, not the capital.",
                    confidence=0.95,
                    evidence=["Australian Bureau of Statistics census figures."]
                )
            ]
            assumptions = ["Focus is on sovereign federal national capital."]
            sources = ["Australian Government Official Registry", "ABS Census Data"]
        elif "solar" in prompt_lower or "jupiter" in prompt_lower or "planet" in prompt_lower:
            answer = "Jupiter is the largest planet in our solar system, with a mass exceeding 300 Earth masses."
            claims = [
                ObservableClaim(
                    claim="Jupiter is the largest planet in the Solar System.",
                    confidence=0.99,
                    evidence=["NASA Planetary Fact Sheet indicates Jupiter planetary diameter of 142,984 km."]
                ),
                ObservableClaim(
                    claim="Jupiter contains more than twice the mass of all other solar planets combined.",
                    confidence=0.96,
                    evidence=["Planetary orbital mechanics and astronomical observation."]
                )
            ]
            assumptions = ["Considers currently defined IAU planets."]
            sources = ["NASA Planetary Database"]
        else:
            # General factual template
            answer = f"Synthesized findings addressing query: verified structured response based on analytical ground truth."
            claims = [
                ObservableClaim(
                    claim=f"Primary assertion regarding {prompt[:30].strip()}",
                    confidence=round(rng.uniform(0.85, 0.98), 2),
                    evidence=[f"Context documentation confirms properties of {prompt[:20].strip()}."]
                ),
                ObservableClaim(
                    claim=f"Secondary supporting correlation with verified validation metrics.",
                    confidence=round(rng.uniform(0.80, 0.92), 2),
                    evidence=["Cross-referenced against verified reference repository."]
                )
            ]
            assumptions = ["Standard environmental baseline parameters apply."]
            sources = ["Reference Knowledge Repository v1.0"]

        if issubclass(schema, ObservableReasoningArtifact):
            return schema(
                answer=answer,
                claims=claims,
                assumptions=assumptions,
                sources=sources,
            )
        
        # Fallback if another schema is requested
        try:
            return schema(
                answer=answer,
                claims=[c.model_dump() for c in claims],
                assumptions=assumptions,
                sources=sources,
            )
        except Exception:
            return schema.model_validate({"answer": answer})
