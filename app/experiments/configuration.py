"""Experiment configuration dataclass."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExperimentRunConfig(BaseModel):
    experiment_id: str
    seed: int = 42
    models: List[str] = Field(default_factory=lambda: ["mock"])
    architectures: List[str] = Field(default_factory=lambda: ["sequential", "hierarchical", "parallel_consensus"])
    conditions: List[str] = Field(default_factory=lambda: ["INTERCEPTOR_ON", "INTERCEPTOR_OFF"])
    fault_types: List[str] = Field(default_factory=lambda: ["factual_mutation", "numerical_mutation", "contradiction_insertion"])
    num_tasks: int = 15
    use_adaptive_calibration: bool = True
