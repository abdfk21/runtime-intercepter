"""Model calibration profile storage and serialization."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ModelCalibrationProfile(BaseModel):
    """Calibrated operating parameters and empirical metrics for a specific LLM."""
    model_name: str
    calibration_method: str = "grid_search_f1"
    threshold: float = Field(..., ge=0.0, le=1.0)
    warning_threshold: float = Field(0.35, ge=0.0, le=1.0)
    block_threshold: float = Field(0.80, ge=0.0, le=1.0)
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "schema_weight": 0.25,
            "reasoning_weight": 0.40,
            "consistency_weight": 0.35,
        }
    )
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    false_positive_rate: float = Field(..., ge=0.0, le=1.0)
    false_negative_rate: float = Field(..., ge=0.0, le=1.0)
    validation_size: int = Field(..., gt=0)
    dataset_version: str = "v1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        # Safe filename
        safe_name = self.model_name.replace("/", "_").replace(":", "_")
        filepath = directory / f"calibration_{safe_name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return filepath

    @classmethod
    def load(cls, filepath: Path) -> "ModelCalibrationProfile":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
