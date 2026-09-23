"""Standalone script: Run benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli import benchmark

if __name__ == "__main__":
    benchmark(num_tasks=12)
