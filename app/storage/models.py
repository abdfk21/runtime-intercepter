"""SQLAlchemy database models for persistence."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship
from app.storage.database import Base


class ExperimentDB(Base):
    __tablename__ = "experiments"

    experiment_id = Column(String(64), primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_name = Column(String(64), nullable=False)
    architecture = Column(String(64), nullable=False)
    condition = Column(String(32), nullable=False)  # INTERCEPTOR_ON, INTERCEPTOR_OFF
    random_seed = Column(Integer, default=42)
    dataset_version = Column(String(32), default="v1.0")
    config_json = Column(JSON, default=dict)
    status = Column(String(32), default="COMPLETED")

    traces = relationship("TraceDB", back_populates="experiment", cascade="all, delete-orphan")
    results = relationship("ExperimentResultDB", back_populates="experiment", cascade="all, delete-orphan")


class TaskDB(Base):
    __tablename__ = "tasks"

    task_id = Column(String(64), primary_key=True, index=True)
    category = Column(String(64), nullable=False)
    prompt = Column(Text, nullable=False)
    difficulty = Column(String(32), default="medium")
    ground_truth = Column(Text, nullable=False)
    reference_facts = Column(JSON, default=list)
    metadata_json = Column(JSON, default=dict)

    traces = relationship("TraceDB", back_populates="task")


class TraceDB(Base):
    __tablename__ = "traces"

    trace_id = Column(String(64), primary_key=True, index=True)
    experiment_id = Column(String(64), ForeignKey("experiments.experiment_id"), nullable=True)
    task_id = Column(String(64), ForeignKey("tasks.task_id"), nullable=True)
    model = Column(String(64), nullable=False)
    architecture = Column(String(64), nullable=False)
    condition = Column(String(32), nullable=False)
    fault_injected = Column(Boolean, default=False)
    fault_type = Column(String(64), nullable=True)
    cascade_occurred = Column(Boolean, default=False)
    cascade_depth = Column(Integer, default=0)
    cascade_severity = Column(Float, default=0.0)
    error_contained = Column(Boolean, default=False)
    total_latency_ms = Column(Float, default=0.0)
    interceptor_latency_ms = Column(Float, default=0.0)
    repair_latency_ms = Column(Float, default=0.0)
    final_output = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    experiment = relationship("ExperimentDB", back_populates="traces")
    task = relationship("TaskDB", back_populates="traces")
    handoffs = relationship("HandoffDB", back_populates="trace", cascade="all, delete-orphan")
    fault = relationship("FaultInjectionDB", back_populates="trace", uselist=False, cascade="all, delete-orphan")


class HandoffDB(Base):
    __tablename__ = "handoffs"

    handoff_id = Column(String(64), primary_key=True, index=True)
    trace_id = Column(String(64), ForeignKey("traces.trace_id"), nullable=False)
    source_agent = Column(String(64), nullable=False)
    target_agent = Column(String(64), nullable=False)
    decision = Column(String(32), nullable=False)
    risk_score = Column(Float, nullable=False)
    schema_score = Column(Float, nullable=False)
    reasoning_score = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    latency_ms = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    trace = relationship("TraceDB", back_populates="handoffs")
    detections = relationship("DetectionDB", back_populates="handoff", cascade="all, delete-orphan")
    repair = relationship("RepairDB", back_populates="handoff", uselist=False, cascade="all, delete-orphan")


class DetectionDB(Base):
    __tablename__ = "detections"

    detection_id = Column(Integer, primary_key=True, autoincrement=True)
    handoff_id = Column(String(64), ForeignKey("handoffs.handoff_id"), nullable=False)
    signal_name = Column(String(64), nullable=False)
    raw_score = Column(Float, nullable=False)
    normalized_score = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    details = Column(JSON, default=dict)
    explanations = Column(JSON, default=list)

    handoff = relationship("HandoffDB", back_populates="detections")


class FaultInjectionDB(Base):
    __tablename__ = "fault_injections"

    fault_id = Column(String(64), primary_key=True, index=True)
    trace_id = Column(String(64), ForeignKey("traces.trace_id"), nullable=False)
    fault_type = Column(String(64), nullable=False)
    location = Column(String(64), nullable=False)
    severity = Column(Float, default=0.8)
    ground_truth = Column(Text, nullable=False)
    random_seed = Column(Integer, default=42)
    original_output = Column(JSON, default=dict)
    corrupted_output = Column(JSON, default=dict)

    trace = relationship("TraceDB", back_populates="fault")


class RepairDB(Base):
    __tablename__ = "repairs"

    repair_id = Column(Integer, primary_key=True, autoincrement=True)
    handoff_id = Column(String(64), ForeignKey("handoffs.handoff_id"), nullable=False)
    attempts = Column(Integer, default=1)
    repair_success = Column(Boolean, default=False)
    original_output = Column(JSON, default=dict)
    repaired_output = Column(JSON, default=dict)
    repair_prompt = Column(Text, nullable=True)

    handoff = relationship("HandoffDB", back_populates="repair")


class AnnotationDB(Base):
    __tablename__ = "annotations"

    annotation_id = Column(String(64), primary_key=True, index=True)
    trace_id = Column(String(64), nullable=False, index=True)
    handoff_id = Column(String(64), nullable=True)
    annotator_id = Column(String(64), nullable=False)
    label = Column(String(32), nullable=False)
    is_synthetic = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CalibrationProfileDB(Base):
    __tablename__ = "calibration_profiles"

    profile_id = Column(String(64), primary_key=True, index=True)
    model_name = Column(String(64), nullable=False, unique=True)
    threshold = Column(Float, nullable=False)
    warning_threshold = Column(Float, nullable=False)
    block_threshold = Column(Float, nullable=False)
    weights = Column(JSON, default=dict)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1 = Column(Float, nullable=False)
    validation_size = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ExperimentResultDB(Base):
    __tablename__ = "experiment_results"

    result_id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), ForeignKey("experiments.experiment_id"), nullable=False)
    metric_name = Column(String(64), nullable=False)
    metric_value = Column(Float, nullable=False)
    metadata_json = Column(JSON, default=dict)

    experiment = relationship("ExperimentDB", back_populates="results")
