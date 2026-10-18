"""FastAPI REST routes for multi-agent interceptor system."""

import os
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
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

    # Attach live provider metadata
    res["model"] = prov_meta["model_name"]
    res["model_name"] = prov_meta["model_name"]
    res["requested_provider"] = prov_meta["requested_provider"]
    res["actual_provider"] = prov_meta["actual_provider"]
    res["fallback_used"] = prov_meta["fallback_used"]
    res["fallback_reason"] = prov_meta.get("fallback_reason")
    res["is_live_api"] = (prov_meta["actual_provider"] in ["gemini", "groq", "ollama"])
    res["prompt"] = req.prompt

    repo = Repository(db)
    # Persist the task and trace for full live inspection
    repo.save_task({
        "task_id": res["task_id"],
        "category": "live_user_task",
        "prompt": req.prompt,
        "difficulty": "medium",
        "ground_truth": "",
        "reference_facts": [],
        "metadata": {
            "model": prov_meta["model_name"],
            "provider": prov_meta["actual_provider"],
            "is_live_api": res["is_live_api"],
        },
    })
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


@router.get("/traces")
def list_traces(
    limit: int = Query(50, ge=1, le=500),
    model_filter: str = Query("all", description="'all', 'realtime', or 'mock'"),
    db: Session = Depends(get_db),
):
    repo = Repository(db)
    all_traces = repo.list_traces(limit=limit * 2)
    output = []

    for t in all_traces:
        is_live = (t.model != "deterministic-mock-v1")
        if model_filter == "realtime" and not is_live:
            continue
        if model_filter == "mock" and is_live:
            continue

        prompt_str = t.task.prompt if t.task and t.task.prompt else ""
        final_answer = ""
        if isinstance(t.final_output, dict):
            final_answer = t.final_output.get("answer", "")
        elif isinstance(t.final_output, str):
            final_answer = t.final_output

        output.append({
            "trace_id": t.trace_id,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "model": t.model,
            "model_name": t.model,
            "architecture": t.architecture,
            "condition": t.condition,
            "fault_injected": t.fault_injected,
            "fault_type": t.fault_type,
            "cascade_occurred": t.cascade_occurred,
            "error_contained": t.error_contained,
            "total_latency_ms": t.total_latency_ms,
            "interceptor_latency_ms": t.interceptor_latency_ms,
            "prompt": prompt_str,
            "final_answer": final_answer,
            "is_live_api": is_live,
            "handoffs_count": len(t.handoffs),
        })

        if len(output) >= limit:
            break

    return output


@router.get("/metrics/summary")
def get_metrics_summary(db: Session = Depends(get_db)):
    repo = Repository(db)
    traces = repo.list_traces(limit=500)

    total = len(traces)
    live_traces = [t for t in traces if t.model != "deterministic-mock-v1"]
    mock_traces = [t for t in traces if t.model == "deterministic-mock-v1"]

    def calc_stats(group):
        if not group:
            return {
                "count": 0,
                "injected_count": 0,
                "containment_rate": 0.0,
                "cascade_rate": 0.0,
                "avg_latency_ms": 0.0,
            }
        injected = [t for t in group if t.fault_injected]
        contained = [t for t in injected if t.error_contained]
        cascaded = [t for t in injected if t.cascade_occurred]
        c_rate = (len(contained) / len(injected) * 100) if injected else 100.0
        casc_rate = (len(cascaded) / len(injected) * 100) if injected else 0.0
        avg_lat = sum(t.total_latency_ms for t in group) / len(group)
        return {
            "count": len(group),
            "injected_count": len(injected),
            "containment_rate": round(c_rate, 1),
            "cascade_rate": round(casc_rate, 1),
            "avg_latency_ms": round(avg_lat, 1),
        }

    # Per-model breakdown
    model_groups = {}
    for t in traces:
        model_groups.setdefault(t.model, []).append(t)
    models_stats = {m: calc_stats(grp) for m, grp in model_groups.items()}

    # Per-architecture breakdown
    arch_groups = {}
    for t in traces:
        arch_groups.setdefault(t.architecture, []).append(t)
    arch_stats = {a: calc_stats(grp) for a, grp in arch_groups.items()}

    gemini_key = bool(os.getenv("GEMINI_API_KEY"))
    groq_key = bool(os.getenv("GROQ_API_KEY"))

    return {
        "total_traces": total,
        "has_live_traces": len(live_traces) > 0,
        "all": calc_stats(traces),
        "live": calc_stats(live_traces),
        "mock": calc_stats(mock_traces),
        "models": models_stats,
        "architectures": arch_stats,
        "providers_status": {
            "gemini": {
                "configured": gemini_key,
                "model": os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
                "status": "online" if gemini_key else "missing_key",
            },
            "groq": {
                "configured": groq_key,
                "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                "status": "online" if groq_key else "missing_key",
            },
            "mock": {
                "configured": True,
                "model": "deterministic-mock-v1",
                "status": "online",
            },
        },
    }


@router.get("/traces/{trace_id}")
def get_trace_details(trace_id: str, db: Session = Depends(get_db)):
    repo = Repository(db)
    trace = repo.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    prompt_str = trace.task.prompt if trace.task and trace.task.prompt else "Multi-Agent System Execution"
    is_live = (trace.model != "deterministic-mock-v1")

    handoffs_list = []
    agent_outputs = {}
    for h in trace.handoffs:
        explanations = []
        for det in getattr(h, "detections", []):
            if det.explanations:
                explanations.extend(det.explanations)

        handoffs_list.append({
            "handoff_id": h.handoff_id,
            "source_agent": h.source_agent,
            "target_agent": h.target_agent,
            "decision": h.decision,
            "risk_score": h.risk_score,
            "schema_score": h.schema_score,
            "reasoning_score": h.reasoning_score,
            "consistency_score": h.consistency_score,
            "threshold": h.threshold,
            "latency_ms": h.latency_ms,
            "explanations": explanations,
        })

    return {
        "trace_id": trace.trace_id,
        "prompt": prompt_str,
        "model": trace.model,
        "model_name": trace.model,
        "architecture": trace.architecture,
        "condition": trace.condition,
        "fault_injected": trace.fault_injected,
        "fault_type": trace.fault_type,
        "cascade_occurred": trace.cascade_occurred,
        "cascade_depth": trace.cascade_depth,
        "cascade_severity": trace.cascade_severity,
        "error_contained": trace.error_contained,
        "total_latency_ms": trace.total_latency_ms,
        "interceptor_latency_ms": trace.interceptor_latency_ms,
        "repair_latency_ms": trace.repair_latency_ms,
        "is_live_api": is_live,
        "final_output": trace.final_output,
        "agent_outputs": agent_outputs,
        "handoffs": handoffs_list,
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
