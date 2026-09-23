"""Application configuration and settings management."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "config"


class ModelsConfig(BaseModel):
    models: Dict[str, Any] = Field(default_factory=dict)
    default_provider: str = "mock"
    fallback_chain: List[str] = Field(default_factory=lambda: ["mock"])


class ArchitecturesConfig(BaseModel):
    architectures: Dict[str, Any] = Field(default_factory=dict)


class DetectionConfig(BaseModel):
    enabled: bool = True
    interceptor_mode: str = "active"
    weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "schema_weight": 0.25,
            "reasoning_weight": 0.40,
            "consistency_weight": 0.35,
        }
    )
    default_threshold: float = 0.50
    warning_threshold: float = 0.35
    block_threshold: float = 0.80
    schema_validator: Dict[str, Any] = Field(default_factory=dict)
    reasoning_monitor: Dict[str, Any] = Field(default_factory=dict)
    nli_checker: Dict[str, Any] = Field(default_factory=dict)
    repair: Dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    enabled: bool = True
    optimization_target: str = "f1"
    min_precision_constraint: float = 0.65
    threshold_search_grid: Dict[str, float] = Field(default_factory=dict)
    weight_search_grid: Dict[str, Any] = Field(default_factory=dict)
    split: Dict[str, Any] = Field(default_factory=dict)
    profiles_dir: str = "data/calibration_profiles"


class ExperimentsConfig(BaseModel):
    random_seed: int = 42
    validation_size: int = 350
    alpha: float = 0.05
    bootstrap_iterations: int = 1000
    evaluation_conditions: List[str] = Field(
        default_factory=lambda: ["INTERCEPTOR_ON", "INTERCEPTOR_OFF"]
    )
    architectures_to_evaluate: List[str] = Field(
        default_factory=lambda: ["sequential", "hierarchical", "parallel_consensus"]
    )
    models_to_evaluate: List[str] = Field(
        default_factory=lambda: ["mock", "gemini", "groq", "ollama"]
    )
    fault_types: List[str] = Field(default_factory=list)
    task_categories: List[str] = Field(default_factory=list)
    export_formats: List[str] = Field(
        default_factory=lambda: ["csv", "markdown", "latex", "json"]
    )


def load_yaml(file_path: Path) -> Dict[str, Any]:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_dir = CONFIG_DIR
        self.data_dir = BASE_DIR / "data"
        self.results_dir = BASE_DIR / "results"

        # Ensure core directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "validation").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "calibration_profiles").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "reports").mkdir(parents=True, exist_ok=True)

        # Environment settings
        self.app_host: str = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port: int = int(os.getenv("APP_PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.database_url: str = os.getenv(
            "DATABASE_URL", f"sqlite:///{self.data_dir / 'interceptor.db'}"
        )
        self.default_provider: str = os.getenv("DEFAULT_PROVIDER", "mock")
        self.random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

        # YAML configs
        self.models_cfg = ModelsConfig(**load_yaml(CONFIG_DIR / "models.yaml"))
        self.architectures_cfg = ArchitecturesConfig(
            **load_yaml(CONFIG_DIR / "architectures.yaml")
        )
        self.detection_cfg = DetectionConfig(
            **load_yaml(CONFIG_DIR / "detection.yaml")
        )
        self.calibration_cfg = CalibrationConfig(
            **load_yaml(CONFIG_DIR / "calibration.yaml")
        )
        self.experiments_cfg = ExperimentsConfig(
            **load_yaml(CONFIG_DIR / "experiments.yaml")
        )


settings = Settings()
