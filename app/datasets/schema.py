"""Schemas for benchmark tasks, execution traces, and human annotations."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    FACTUAL_QA = "factual_qa"
    SUMMARIZATION = "summarization"
    PLANNING = "planning"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    INFORMATION_EXTRACTION = "information_extraction"
    COMPARISON = "comparison"
    NUMERICAL_REASONING = "numerical_reasoning"
    RESEARCH_SYNTHESIS = "research_synthesis"


class AnnotationLabel(str, Enum):
    VALID = "VALID"
    SUSPICIOUS = "SUSPICIOUS"
    HALLUCINATED = "HALLUCINATED"
    CONTRADICTORY = "CONTRADICTORY"


class BenchmarkTask(BaseModel):
    """A benchmark task instance."""
    task_id: str
    category: TaskCategory
    prompt: str
    difficulty: str = "medium"  # easy, medium, hard
    ground_truth: str
    reference_facts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    """Comprehensive trace record of an end-to-end multi-agent execution."""
    trace_id: str
    task_id: str
    architecture: str
    model: str
    condition: str  # "INTERCEPTOR_ON" or "INTERCEPTOR_OFF"
    prompt: str
    agent_outputs: Dict[str, Any]
    handoffs: List[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]
    fault_injected: bool = False
    fault_type: Optional[str] = None
    injected_fault_details: Optional[Dict[str, Any]] = None
    cascade_occurred: bool = False
    cascade_depth: int = 0
    cascade_severity: float = 0.0
    error_contained: bool = False
    total_latency_ms: float = 0.0
    interceptor_latency_ms: float = 0.0
    repair_latency_ms: float = 0.0
    dataset_version: str = "v1.0"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HumanAnnotation(BaseModel):
    """Human label for a trace or handoff."""
    annotation_id: str
    trace_id: str
    handoff_id: Optional[str] = None
    annotator_id: str
    label: AnnotationLabel
    is_synthetic_development_label: bool = False
    notes: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    annotation_version: str = "v1.0"
