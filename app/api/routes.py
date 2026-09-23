"""FastAPI REST routes for multi-agent interceptor system."""

import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.config import settings
from app.api.schemas import (
    HealthResponse,
    TaskRunRequest,
    InterceptRequest,
    ExperimentCreateRequest,
    CalibrationRequest,
    AnnotationCreateRequest,
)
from app.storage.database import get_db
from app.storage.repository import Repository
from app.interceptor.interceptor import RuntimeInterceptor
from app.interceptor.calibration.calibrator import ModelCalibrator
from app.llm.factory import LLMFactory
from app.orchestration.sequential import SequentialPipeline
from app.orchestration.hierarchical import HierarchicalPipeline
from app.orchestration.parallel_consensus import ParallelConsensusPipeline
from app.fault_injection.injector import FaultInjector
from app.fault_injection.fault_types import FaultType
from app.datasets.generator import DatasetGenerator
from app.datasets.annotation import AnnotationManager
from app.evaluation.ablation import AblationStudyRunner

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment="academic_research_v1",
        default_provider=settings.default_provider,
    )


@router.post("/run/task")
def run_task(req: TaskRunRequest, db: Session = Depends(get_db)):
    llm, prov_meta = LLMFactory.create(req.model, allow_fallback=True)
    interceptor = RuntimeInterceptor(mode="active" if req.interceptor_enabled else "passive")
    injector = FaultInjector(seed=42)

    pipelines = {
        "sequential": SequentialPipeline(llm, interceptor, injector),
        "hierarchical": HierarchicalPipeline(llm, interceptor, injector),
        "parallel_consensus": ParallelConsensusPipeline(llm, interceptor, injector),
    }

    pipeline = pipelines.get(req.architecture)
    if not pipeline:
        raise HTTPException(status_code=400, detail=f"Invalid architecture: {req.architecture}")

    fi_cfg = None
    if req.inject_fault:
        fi_cfg = {
            "fault_type": req.fault_type or "factual_mutation",
            "location": "researcher" if req.architecture != "parallel_consensus" else "researcher_a",
            "seed": 42,
        }

    res = pipeline.run(
        prompt=req.prompt,
        interceptor_enabled=req.interceptor_enabled,
        fault_injection_config=fi_cfg,
    )

    repo = Repository(db)
    repo.save_trace(res)
    return res


@router.post("/run/pipeline")
def run_pipeline(req: TaskRunRequest, db: Session = Depends(get_db)):
    return run_task(req, db)


@router.post("/intercept")
def intercept_handoff(req: InterceptRequest):
    interceptor = RuntimeInterceptor()
    res = interceptor.inspect_handoff(
        trace_id=req.trace_id,
        source_agent=req.source_agent,
        target_agent=req.target_agent,
        source_output=req.source_output,
        target_context=req.target_context,
        model_name=req.model_name,
        architecture=req.architecture,
    )
    return res.model_dump()


@router.post("/calibrate")
def run_calibration(req: CalibrationRequest):
    calibrator = ModelCalibrator()
    dataset_gen = DatasetGenerator(seed=42)
    tasks = dataset_gen.generate_tasks(count=req.sample_size)
    traces = dataset_gen.generate_validation_dataset(tasks, target_size=req.sample_size)

    profile = calibrator.calibrate(
        model_name=req.model_name,
        calibration_traces=traces,
        optimization_target=req.target_metric,
        min_precision=req.min_precision,
    )
    return profile.model_dump()


@router.get("/calibration/{model}")
def get_calibration(model: str):
    calibrator = ModelCalibrator()
    profile = calibrator.load_profile(model)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Calibration profile for '{model}' not found.")
    return profile.model_dump()


@router.post("/ablation")
def run_ablation():
    dataset_gen = DatasetGenerator(seed=42)
    tasks = dataset_gen.generate_tasks(count=40)
    traces = dataset_gen.generate_validation_dataset(tasks, target_size=40)
    runner = AblationStudyRunner()
    results = runner.run_ablation(traces)
    return results


@router.get("/traces/{trace_id}")
def get_trace_details(trace_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    trace = repo.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    
    return {
        "trace_id": trace.trace_id,
        "model": trace.model,
        "architecture": trace.architecture,
        "condition": trace.condition,
        "fault_injected": trace.fault_injected,
        "cascade_occurred": trace.cascade_occurred,
        "error_contained": trace.error_contained,
        "total_latency_ms": trace.total_latency_ms,
        "final_output": trace.final_output,
        "handoffs": [
            {
                "handoff_id": h.handoff_id,
                "source_agent": h.source_agent,
                "target_agent": h.target_agent,
                "decision": h.decision,
                "risk_score": h.risk_score,
                "schema_score": h.schema_score,
                "reasoning_score": h.reasoning_score,
                "consistency_score": h.consistency_score,
                "threshold": h.threshold,
            }
            for h in trace.handoffs
        ],
    }


@router.post("/annotations")
def create_annotation(req: AnnotationCreateRequest, db: Session = Depends(get_db)):
    mgr = AnnotationManager()
    ann = mgr.add_annotation(
        trace_id=req.trace_id,
        annotator_id=req.annotator_id,
        label=req.label,
        handoff_id=req.handoff_id,
        notes=req.notes,
        is_synthetic=False,
    )
    repo = Repository(db)
    repo.save_annotation(ann.model_dump())
    return ann.model_dump()


@router.get("/annotations")
def list_annotations():
    mgr = AnnotationManager()
    return [a.model_dump() for a in mgr.annotations]
