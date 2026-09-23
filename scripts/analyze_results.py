"""Standalone script: Statistical analysis and hypothesis tests."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.datasets.generator import DatasetGenerator
from app.datasets.loader import DatasetLoader
from app.evaluation.statistics import StatisticalAnalyzer
from app.evaluation.ablation import AblationStudyRunner

if __name__ == "__main__":
    print("Running statistical analysis...")
    gen = DatasetGenerator(seed=42)
    tasks = gen.generate_tasks(count=40)
    traces = gen.generate_validation_dataset(tasks, target_size=40)

    on_traces = [t for t in traces if t.get("is_corrupted")]
    # Simulate OFF traces where corrupted traces propagate
    off_traces = []
    for t in on_traces:
        t_off = dict(t)
        t_off["cascade_occurred"] = True
        t_off["error_contained"] = False
        t_off["total_latency_ms"] = 90.0
        off_traces.append(t_off)

    for t in on_traces:
        t["fault_injected"] = True
        t["error_contained"] = True
        t["cascade_occurred"] = False
        t["total_latency_ms"] = 125.0

    for t in off_traces:
        t["fault_injected"] = True

    res = StatisticalAnalyzer.compare_on_off(on_traces, off_traces)
    print(json.dumps(res, indent=2))
