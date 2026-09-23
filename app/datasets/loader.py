"""Dataset loader and stratified partitioner preventing data leakage."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from sklearn.model_selection import train_test_split
from app.core.config import settings
from app.core.logging import logger


class DatasetLoader:
    """Loads and splits benchmark datasets without data leakage."""

    @staticmethod
    def load_traces(filepath: Path) -> List[Dict[str, Any]]:
        if not filepath.exists():
            raise FileNotFoundError(f"Trace dataset not found at {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def split_calibration_test(
        traces: List[Dict[str, Any]],
        calibration_ratio: float = 0.50,
        random_seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Strict stratified split ensuring zero data leakage between calibration and test sets."""
        labels = [1 if t.get("is_corrupted", False) else 0 for t in traces]
        
        calib_traces, test_traces = train_test_split(
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
