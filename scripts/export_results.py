"""Standalone script: Export figures and research tables."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli import full_experiment

if __name__ == "__main__":
    full_experiment()
