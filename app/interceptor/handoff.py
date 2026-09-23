"""Data models for agent handoffs, detection signals, and interceptor decisions."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InterceptorDecision(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNING = "PASS_WITH_WARNING"
    REPAIR = "REPAIR"
    BLOCK = "BLOCK"


class SignalDetails(BaseModel):
    """Output details from an individual detection signal."""
    signal_name: str
    raw_score: float = Field(..., ge=0.0)
    normalized_score: float = Field(..., ge=0.0, le=1.0)
    weight: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    details: Dict[str, Any] = Field(default_factory=dict)
    explanations: List[str] = Field(default_factory=list)


class HandoffContext(BaseModel):
    """Contextual metadata surrounding an agent handoff."""
    trace_id: str
    handoff_id: str
    source_agent: str
    target_agent: str
    model_name: str
    architecture: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HandoffInspectionResult(BaseModel):
    """Comprehensive result of an interceptor handoff inspection."""
    trace_id: str
    handoff_id: str
    source_agent: str = ""
    target_agent: str = ""
    decision: InterceptorDecision
    risk_score: float = Field(..., ge=0.0, le=1.0)
    schema_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_score: float = Field(..., ge=0.0, le=1.0)
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    signals: Dict[str, SignalDetails] = Field(default_factory=dict)
    explanations: List[str] = Field(default_factory=list)
    repair_required: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = 0.0
    repair_attempt: int = 0
    repaired_output: Optional[Dict[str, Any]] = None
