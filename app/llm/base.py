"""Base LLM Provider interface and structured reasoning models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field


class ObservableClaim(BaseModel):
    """An observable factual claim extracted from reasoning."""
    claim: str = Field(..., description="The factual assertion made by the agent")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Self-reported confidence in the claim"
    )
    evidence: List[str] = Field(
        default_factory=list, description="Observable grounded evidence supporting claim"
    )


class ObservableReasoningArtifact(BaseModel):
    """Observable structured reasoning artifact.
    
    CRITICAL: Does NOT use or expose hidden chain-of-thought.
    All claims, assumptions, sources, and evidence are fully observable.
    """
    answer: str = Field(..., description="The primary response or synthesis")
    claims: List[ObservableClaim] = Field(
        default_factory=list, description="List of atomic claims with confidence and evidence"
    )
    assumptions: List[str] = Field(
        default_factory=list, description="Explicit assumptions stated by the agent"
    )
    sources: List[str] = Field(
        default_factory=list, description="Explicit sources or citations referenced"
    )


class BaseLLMProvider(ABC):
    """Common interface for all LLM providers."""

    def __init__(self, model_name: str, temperature: float = 0.2, max_tokens: int = 1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a natural language completion."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel] = ObservableReasoningArtifact,
        system_prompt: Optional[str] = None,
    ) -> BaseModel:
        """Generate an output strictly validated against a Pydantic schema."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the identifier for the active model."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify provider availability and valid credentials."""
        pass
