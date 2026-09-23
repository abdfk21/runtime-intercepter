"""Standalone script: Calibrate models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli import calibrate

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "deterministic-mock-v1"
    calibrate(model_name=model)
