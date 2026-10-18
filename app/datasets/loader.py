"""Dataset loader and stratified partitioner preventing data leakage."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.core.ml_compat import safe_train_test_split
from app.datasets.schema import BenchmarkTask, TaskCategory


class DatasetLoader:
    """Loads and splits benchmark datasets without data leakage."""

    @staticmethod
    def load_traces(filepath: Path) -> List[Dict[str, Any]]:
        if not filepath.exists():
            raise FileNotFoundError(f"Trace dataset not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_converted_external_datasets() -> List[Dict[str, Any]]:
        """Load all converted external datasets if they exist.

        This supports the newly added synthetic/bootstrap conversion outputs under
        `data/processed/converted_external_datasets` and merges them into a single list.
        """
        processed_dir = settings.data_dir / "processed" / "converted_external_datasets"
        if not processed_dir.exists():
            return []

        traces: List[Dict[str, Any]] = []
        for path in sorted(processed_dir.glob("*_converted.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    traces.extend(payload)
            except (json.JSONDecodeError, OSError):
                continue
        return traces

    @staticmethod
    def load_converted_external_tasks() -> List[BenchmarkTask]:
        """Convert external trace records into BenchmarkTask-like task objects.

        This keeps the main benchmark pipeline working with the converted external datasets
        without requiring a separate command or custom loader in each runner.
        """
        traces = DatasetLoader.load_converted_external_datasets()
        tasks: List[BenchmarkTask] = []

        for idx, trace in enumerate(traces):
            prompt = str(trace.get("prompt") or "").strip()
            if not prompt:
                continue

            category_name = str(trace.get("task_category") or "factual_qa")
            try:
                category = TaskCategory(category_name)
            except ValueError:
                category = TaskCategory.FACTUAL_QA

            ground_truth = str(
                trace.get("metadata", {}).get("raw_record", {}).get("ground_truth")
                or trace.get("metadata", {}).get("raw_record", {}).get("answer")
                or trace.get("final_output", {}).get("answer")
                or trace.get("source_output", {}).get("answer")
                or "Converted external benchmark task."
            ).strip()

            tasks.append(
                BenchmarkTask(
                    task_id=str(trace.get("task_id") or f"converted_task_{idx:04d}"),
                    category=category,
                    prompt=prompt,
                    difficulty="medium",
                    ground_truth=ground_truth,
                    reference_facts=[ground_truth],
                    metadata={
                        "source": "converted_external_dataset",
                        "trace_id": trace.get("trace_id"),
                        "architecture": trace.get("architecture"),
                        "model": trace.get("model"),
                    },
                )
            )

        return tasks

    @staticmethod
    def split_calibration_test(
        traces: List[Dict[str, Any]],
        calibration_ratio: float = 0.50,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Strict stratified split ensuring zero data leakage between calibration and test sets."""
        labels = [1 if t.get("is_corrupted", False) else 0 for t in traces]
        
        calib_traces, test_traces = safe_train_test_split(
            traces,
            train_size=calibration_ratio,
            random_state=random_seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )

        logger.info(
            f"Split dataset of {len(traces)} traces: "
            f"{len(calib_traces)} calibration traces, {len(test_traces)} held-out test traces."
        )
        return calib_traces, test_traces
