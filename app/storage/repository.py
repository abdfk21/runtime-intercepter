"""Storage repository for CRUD operations."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.storage.models import (
    ExperimentDB,
    TaskDB,
    TraceDB,
    HandoffDB,
    DetectionDB,
    FaultInjectionDB,
    RepairDB,
    AnnotationDB,
    CalibrationProfileDB,
    ExperimentResultDB,
)


class Repository:
    """Encapsulates persistence logic for experiment and interceptor data."""

    def __init__(self, db: Session):
        self.db = db

    def save_experiment(
        self,
        experiment_id: str,
        model_name: str,
        architecture: str,
        condition: str,
        random_seed: int = 42,
        config: Dict[str, Any] = None,
    ) -> ExperimentDB:
        exp = ExperimentDB(
            experiment_id=experiment_id,
            model_name=model_name,
            architecture=architecture,
            condition=condition,
            random_seed=random_seed,
            config_json=config or {},
        )
        self.db.merge(exp)
        self.db.commit()
        return exp

    def save_task(self, task_data: Dict[str, Any]) -> TaskDB:
        task = TaskDB(
            task_id=task_data["task_id"],
            category=task_data["category"],
            prompt=task_data["prompt"],
            difficulty=task_data.get("difficulty", "medium"),
            ground_truth=task_data.get("ground_truth", ""),
            reference_facts=task_data.get("reference_facts", []),
            metadata_json=task_data.get("metadata", {}),
        )
        self.db.merge(task)
        self.db.commit()
        return task

    def save_trace(self, trace_data: Dict[str, Any]) -> TraceDB:
        trace = TraceDB(
            trace_id=trace_data["trace_id"],
            experiment_id=trace_data.get("experiment_id"),
            task_id=trace_data.get("task_id"),
            model=trace_data["model_name"],
            architecture=trace_data["architecture"],
            condition="INTERCEPTOR_ON" if trace_data.get("interceptor_enabled") else "INTERCEPTOR_OFF",
            fault_injected=trace_data.get("fault_injected", False),
            fault_type=trace_data.get("fault_type"),
            cascade_occurred=trace_data.get("cascade_occurred", False),
            cascade_depth=trace_data.get("cascade_depth", 0),
            cascade_severity=trace_data.get("cascade_severity", 0.0),
            error_contained=trace_data.get("error_contained", False),
            total_latency_ms=trace_data.get("total_latency_ms", 0.0),
            interceptor_latency_ms=trace_data.get("interceptor_latency_ms", 0.0),
            repair_latency_ms=trace_data.get("repair_latency_ms", 0.0),
            final_output=trace_data.get("final_output"),
        )
        self.db.merge(trace)

        # Save handoffs
        for h in trace_data.get("handoffs", []):
            handoff = HandoffDB(
                handoff_id=h["handoff_id"],
                trace_id=trace_data["trace_id"],
                source_agent=h.get("source_agent", "unknown"),
                target_agent=h.get("target_agent", "unknown"),
                decision=h["decision"],
                risk_score=h["risk_score"],
                schema_score=h["schema_score"],
                reasoning_score=h["reasoning_score"],
                consistency_score=h["consistency_score"],
                threshold=h["threshold"],
                latency_ms=h.get("latency_ms", 0.0),
            )
            self.db.merge(handoff)

            # Save signals
            for sig_name, sig_obj in h.get("signals", {}).items():
                det = DetectionDB(
                    handoff_id=h["handoff_id"],
                    signal_name=sig_name,
                    raw_score=sig_obj.get("raw_score", 0.0),
                    normalized_score=sig_obj.get("normalized_score", 0.0),
                    passed=sig_obj.get("passed", True),
                    details=sig_obj.get("details", {}),
                    explanations=sig_obj.get("explanations", []),
                )
                self.db.add(det)

        # Save fault injection
        if trace_data.get("injected_fault_details"):
            fi = trace_data["injected_fault_details"]
            f_db = FaultInjectionDB(
                fault_id=fi.get("fault_id", "flt_0"),
                trace_id=trace_data["trace_id"],
                fault_type=fi.get("fault_type", "factual_mutation"),
                location=fi.get("location", "agent_handoff"),
                severity=fi.get("severity", 0.8),
                ground_truth=fi.get("ground_truth", ""),
                random_seed=fi.get("random_seed", 42),
                original_output=fi.get("original_output", {}),
                corrupted_output=fi.get("corrupted_output", {}),
            )
            self.db.merge(f_db)

        self.db.commit()
        return trace

    def save_annotation(self, ann: Dict[str, Any]) -> AnnotationDB:
        db_ann = AnnotationDB(
            annotation_id=ann["annotation_id"],
            trace_id=ann["trace_id"],
            handoff_id=ann.get("handoff_id"),
            annotator_id=ann["annotator_id"],
            label=ann["label"],
            is_synthetic=ann.get("is_synthetic_development_label", False),
            notes=ann.get("notes"),
        )
        self.db.merge(db_ann)
        self.db.commit()
        return db_ann

    def list_experiments(self) -> List[ExperimentDB]:
        return self.db.query(ExperimentDB).all()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentDB]:
        return self.db.query(ExperimentDB).filter(ExperimentDB.experiment_id == experiment_id).first()

    def list_traces(self, limit: int = 100) -> List[TraceDB]:
        return self.db.query(TraceDB).order_by(TraceDB.created_at.desc()).limit(limit).all()

    def get_trace(self, trace_id: str) -> Optional[TraceDB]:
        return self.db.query(TraceDB).filter(TraceDB.trace_id == trace_id).first()
