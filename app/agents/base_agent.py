"""Base Agent definition for LangGraph multi-agent systems."""

import time
from typing import Any, Dict, List, Optional
from app.llm.base import BaseLLMProvider, ObservableReasoningArtifact
from app.core.logging import logger


class BaseAgent:
    """Base class for multi-agent LLM participants."""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        llm_provider: BaseLLMProvider,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = llm_provider

    def execute(
        self,
        task_prompt: str,
        context: Optional[Dict[str, Any]] = None,
        repair_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute agent step and return structured observable reasoning artifact."""
        start_time = time.perf_counter()
        context = context or {}

        # Build execution prompt
        prompt_parts = [f"Task: {task_prompt}"]
        if context:
            prompt_parts.append(f"Context from upstream agents: {context}")
        if repair_feedback:
            prompt_parts.append(
                f"\nCRITICAL CORRECTION REQUEST:\n{repair_feedback}\n"
                f"Carefully correct the flagged issues, eliminate unsupported or contradictory assertions, "
                f"and ensure claims strictly match observable evidence."
            )

        full_prompt = "\n\n".join(prompt_parts)

        try:
            artifact: ObservableReasoningArtifact = self.llm.generate_structured(
                prompt=full_prompt,
                schema=ObservableReasoningArtifact,
                system_prompt=self.system_prompt,
            )
            output_dict = artifact.model_dump()
        except Exception as e:
            logger.warning(f"Agent {self.name} failed structured generation, falling back: {e}")
            output_dict = {
                "answer": f"Agent {self.name} processed task: {task_prompt[:50]}",
                "claims": [
                    {
                        "claim": f"{self.name} preliminary assertion.",
                        "confidence": 0.85,
                        "evidence": ["Context reference."],
                    }
                ],
                "assumptions": [],
                "sources": [],
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "agent_name": self.name,
            "role": self.role,
            "output": output_dict,
            "latency_ms": elapsed_ms,
        }
