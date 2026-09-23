"""Standalone script: Generate benchmark tasks and traces."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli import generate_dataset_cmd

if __name__ == "__main__":
    generate_dataset_cmd(size=350)
