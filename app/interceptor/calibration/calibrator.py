"""Adaptive per-model calibrator pipeline."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from app.core.config import settings
from app.core.logging import logger
from app.interceptor.calibration.model_profile import ModelCalibrationProfile
from app.interceptor.calibration.threshold_optimizer import ThresholdOptimizer
from app.interceptor.signals.schema_validator import SchemaValidator
from app.interceptor.signals.reasoning_monitor import ReasoningMonitor
from app.interceptor.signals.nli_consistency_checker import NLIConsistencyChecker
from app.interceptor.fusion import SignalFusion


class ModelCalibrator:
    """Calibrates detection thresholds and signal weights independently per LLM."""

    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or (settings.base_dir / settings.calibration_cfg.profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.schema_validator = SchemaValidator()
        self.reasoning_monitor = ReasoningMonitor()
        self.nli_checker = NLIConsistencyChecker()
        self.fusion = SignalFusion()

    def calibrate(
        self,
        model_name: str,
        calibration_traces: List[Dict[str, Any]],
        optimization_target: str = "f1",
        min_precision: float = 0.65,
    ) -> ModelCalibrationProfile:
        """Run calibration on training traces and generate profile."""
        if not calibration_traces:
            raise ValueError("Cannot calibrate on empty trace list.")

        logger.info(f"Starting calibration for model '{model_name}' on {len(calibration_traces)} traces.")

        y_true: List[int] = []
        schema_scores: List[float] = []
        reasoning_scores: List[float] = []
        consistency_scores: List[float] = []

        for trace in calibration_traces:
            # Ground truth label: 1 if corrupted/hallucinated, 0 if clean
            is_corrupted = 1 if trace.get("is_corrupted", False) or trace.get("ground_truth_label") in ["HALLUCINATED", "CORRUPTED", "FAULT"] else 0
            y_true.append(is_corrupted)

            source_output = trace.get("source_output", {})
            target_context = trace.get("target_context", {})

            sig1 = self.schema_validator.validate(source_output)
            sig2 = self.reasoning_monitor.evaluate(source_output, target_context)
            sig3 = self.nli_checker.evaluate(source_output, target_context)

            schema_scores.append(sig1.normalized_score)
            reasoning_scores.append(sig2.normalized_score)
            consistency_scores.append(sig3.normalized_score)

        # Weight optimization (coarse grid search around defaults)
        best_f1 = -1.0
        best_weights = {"schema_weight": 0.25, "reasoning_weight": 0.40, "consistency_weight": 0.35}
        best_opt_res = None

        candidate_weights = [
            (0.20, 0.40, 0.40),
            (0.25, 0.40, 0.35),
            (0.15, 0.50, 0.35),
            (0.30, 0.35, 0.35),
            (0.10, 0.45, 0.45),
        ]

        for sw, rw, cw in candidate_weights:
            self.fusion.update_weights(sw, rw, cw)
            composite_scores = []
            for s, r, c in zip(schema_scores, reasoning_scores, consistency_scores):
                score = (self.fusion.w_schema * s) + (self.fusion.w_reasoning * r) + (self.fusion.w_consistency * c)
                composite_scores.append(score)

            opt_res = ThresholdOptimizer.optimize(
                y_true=y_true,
                scores=composite_scores,
                target_metric=optimization_target,
                min_precision=min_precision,
            )

            metrics = opt_res["best_metrics"]
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_weights = {"schema_weight": sw, "reasoning_weight": rw, "consistency_weight": cw}
                best_opt_res = opt_res

        metrics = best_opt_res["best_metrics"]
        threshold = best_opt_res["best_threshold"]

        profile = ModelCalibrationProfile(
            model_name=model_name,
            calibration_method=f"grid_search_{optimization_target}",
            threshold=threshold,
            warning_threshold=round(max(0.15, threshold * 0.70), 3),
            block_threshold=round(min(0.95, threshold * 1.35), 3),
            weights=best_weights,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            false_positive_rate=metrics["fpr"],
            false_negative_rate=metrics["fnr"],
            validation_size=len(calibration_traces),
            dataset_version="v1.0",
        )

        saved_path = profile.save(self.profiles_dir)
        logger.info(f"Calibrated profile for '{model_name}' saved to {saved_path} (F1: {profile.f1:.4f}, Threshold: {profile.threshold:.3f})")
        return profile

    def load_profile(self, model_name: str) -> Optional[ModelCalibrationProfile]:
        safe_name = model_name.replace("/", "_").replace(":", "_")
        filepath = self.profiles_dir / f"calibration_{safe_name}.json"
        if filepath.exists():
            return ModelCalibrationProfile.load(filepath)
        return None
