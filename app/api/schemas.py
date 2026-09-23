"""Pydantic schemas for REST API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    default_provider: str


class TaskRunRequest(BaseModel):
    prompt: str
    architecture: str = "sequential"
    model: str = "mock"
    interceptor_enabled: bool = True
    inject_fault: bool = False
    fault_type: Optional[str] = "factual_mutation"


class InterceptRequest(BaseModel):
    trace_id: str
    source_agent: str
    target_agent: str
    source_output: Dict[str, Any]
    target_context: Optional[Dict[str, Any]] = None
    model_name: str = "mock"
    architecture: str = "sequential"


class ExperimentCreateRequest(BaseModel):
    model: str = "mock"
    architecture: str = "sequential"
    condition: str = "INTERCEPTOR_ON"
    num_tasks: int = 5
    random_seed: int = 42


class CalibrationRequest(BaseModel):
    model_name: str = "mock"
    target_metric: str = "f1"
    min_precision: float = 0.65
    sample_size: int = 40


class AnnotationCreateRequest(BaseModel):
    trace_id: str
    annotator_id: str
    label: str  # VALID, SUSPICIOUS, HALLUCINATED, CONTRADICTORY
    handoff_id: Optional[str] = None
    notes: Optional[str] = None
