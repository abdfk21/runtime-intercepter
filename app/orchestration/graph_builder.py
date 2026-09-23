"""LangGraph state representation and pipeline execution contexts."""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END


class MultiAgentState(TypedDict):
    """Observable state passed along the multi-agent graph."""
    task_id: str
    prompt: str
    trace_id: str
    model_name: str
    architecture: str
    interceptor_enabled: bool
    fault_injection_config: Optional[Dict[str, Any]]
    
    # Trace history of agent outputs
    agent_outputs: Dict[str, Any]
    
    # Handoff inspection history
    handoffs: List[Dict[str, Any]]
    
    # Containment & error tracking
    fault_injected: bool
    fault_type: Optional[str]
    fault_location: Optional[str]
    injected_fault_details: Optional[Dict[str, Any]]
    
    # Cascade metrics
    cascade_occurred: bool
    cascade_depth: int
    cascade_severity: float
    error_contained: bool
    
    # Final output & latency
    final_output: Optional[Dict[str, Any]]
    total_latency_ms: float
    interceptor_latency_ms: float
    repair_latency_ms: float
