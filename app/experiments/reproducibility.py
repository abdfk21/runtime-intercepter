"""Reproducibility metadata recording: seeds, commit hashes, package versions."""

import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict
import numpy as np


class ReproducibilityTracker:
    """Captures runtime environment and state for academic reproducibility."""

    @staticmethod
    def get_git_commit() -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            return res.stdout.strip()
        except Exception:
            return "git-unavailable-or-uncommitted"

    @classmethod
    def capture_environment_metadata(cls, seed: int = 42) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_seed": seed,
            "git_commit": cls.get_git_commit(),
            "python_version": platform.python_version(),
            "operating_system": platform.platform(),
            "cpu_architecture": platform.machine(),
            "env_default_provider": os.getenv("DEFAULT_PROVIDER", "mock"),
        }

    @staticmethod
    def set_deterministic_seeds(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
