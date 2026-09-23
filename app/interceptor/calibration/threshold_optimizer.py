"""Threshold optimizer for adaptive per-model calibration."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ThresholdOptimizer:
    """Finds optimal decision thresholds and weights to maximize detection performance on calibration data."""

    @staticmethod
    def evaluate_threshold(
        y_true: List[int],
        scores: List[float],
        threshold: float,
    ) -> Dict[str, float]:
        """Compute evaluation metrics at a specific decision threshold."""
        y_true_arr = np.array(y_true, dtype=int)
        scores_arr = np.array(scores, dtype=float)
        y_pred = (scores_arr >= threshold).astype(int)

        tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true_arr == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        accuracy = (tp + tn) / max(1, len(y_true_arr))

        return {
            "threshold": round(threshold, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "accuracy": round(accuracy, 4),
        }

    @classmethod
    def optimize(
        cls,
        y_true: List[int],
        scores: List[float],
        target_metric: str = "f1",
        min_precision: float = 0.60,
        min_threshold: float = 0.15,
        max_threshold: float = 0.85,
        step: float = 0.02,
    ) -> Dict[str, Any]:
        """Search threshold grid for the optimal operating point."""
        best_metrics: Optional[Dict[str, float]] = None
        best_score = -1.0
        search_history = []

        thresholds = np.arange(min_threshold, max_threshold + step / 2, step)
        for t in thresholds:
            t_float = float(round(t, 3))
            metrics = cls.evaluate_threshold(y_true, scores, t_float)
            search_history.append(metrics)

            # Evaluate objective
            score = metrics.get(target_metric, 0.0)
            prec = metrics.get("precision", 0.0)

            # Preference with constraint
            if prec >= min_precision:
                if score > best_score:
                    best_score = score
                    best_metrics = metrics

        # Fallback if no threshold met min_precision constraint
        if best_metrics is None:
            # Pick best F1 regardless of precision
            best_metrics = max(search_history, key=lambda m: m["f1"])

        return {
            "best_threshold": best_metrics["threshold"],
            "best_metrics": best_metrics,
            "search_history": search_history,
        }
