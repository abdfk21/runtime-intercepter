"""Ablation study runner for evaluating individual signal contributions and calibration impacts."""

from typing import Any, Dict, List
from app.interceptor.signals.schema_validator import SchemaValidator
from app.interceptor.signals.reasoning_monitor import ReasoningMonitor
from app.interceptor.signals.nli_consistency_checker import NLIConsistencyChecker
from app.interceptor.calibration.threshold_optimizer import ThresholdOptimizer
from app.evaluation.metrics import EvaluationMetrics


class AblationStudyRunner:
    """Evaluates component-level contributions across signals and calibration modes."""

    def __init__(self):
        self.schema_validator = SchemaValidator()
        self.reasoning_monitor = ReasoningMonitor()
        self.nli_checker = NLIConsistencyChecker()

    def run_ablation(self, test_traces: List[Dict[str, Any]], default_threshold: float = 0.50) -> Dict[str, Any]:
        """Runs full ablation across 9 configurations on test traces."""
        y_true: List[int] = []
        schema_scores: List[float] = []
        reasoning_scores: List[float] = []
        consistency_scores: List[float] = []

        for t in test_traces:
            is_corrupt = 1 if (t.get("is_corrupted", False) or t.get("fault_injected", False)) else 0
            y_true.append(is_corrupt)

            handoffs = t.get("handoffs", [])
            if handoffs:
                s1_val = max(h.get("schema_score", 0.0) for h in handoffs)
                s2_val = max(h.get("reasoning_score", 0.0) for h in handoffs)
                s3_val = max(h.get("consistency_score", 0.0) for h in handoffs)
            else:
                src_out = t.get("source_output") or t.get("agent_outputs", {}).get("researcher") or t.get("final_output") or {}
                tgt_ctx = t.get("target_context") or {"prompt": t.get("prompt", "")}
                if t.get("injected_fault_details") and t["injected_fault_details"].get("ground_truth"):
                    tgt_ctx["ground_truth"] = t["injected_fault_details"]["ground_truth"]

                s1 = self.schema_validator.validate(src_out)
                s2 = self.reasoning_monitor.evaluate(src_out, tgt_ctx)
                s3 = self.nli_checker.evaluate(src_out, tgt_ctx)
                s1_val, s2_val, s3_val = s1.normalized_score, s2.normalized_score, s3.normalized_score

            schema_scores.append(s1_val)
            reasoning_scores.append(s2_val)
            consistency_scores.append(s3_val)

        # Configurations to evaluate: (name, weights: tuple(s, r, c))
        configs = {
            "full_system": (0.25, 0.40, 0.35),
            "schema_only": (1.0, 0.0, 0.0),
            "reasoning_only": (0.0, 1.0, 0.0),
            "nli_only": (0.0, 0.0, 1.0),
            "schema_plus_reasoning": (0.40, 0.60, 0.0),
            "schema_plus_nli": (0.40, 0.0, 0.60),
            "reasoning_plus_nli": (0.0, 0.50, 0.50),
            "no_adaptive_calibration": (0.25, 0.40, 0.35),  # fixed threshold
            "adaptive_calibration": (0.25, 0.40, 0.35),     # tuned threshold
        }

        results: Dict[str, Any] = {}

        for cfg_name, (ws, wr, wc) in configs.items():
            tot = ws + wr + wc
            ws_n, wr_n, wc_n = ws / tot, wr / tot, wc / tot

            composite = [
                (ws_n * s) + (wr_n * r) + (wc_n * c)
                for s, r, c in zip(schema_scores, reasoning_scores, consistency_scores)
            ]

            if cfg_name == "no_adaptive_calibration":
                th = default_threshold
            elif cfg_name == "adaptive_calibration":
                # Find optimal on this distribution
                opt = ThresholdOptimizer.optimize(y_true, composite, target_metric="f1")
                th = opt["best_threshold"]
            else:
                opt = ThresholdOptimizer.optimize(y_true, composite, target_metric="f1")
                th = opt["best_threshold"]

            y_pred = [1 if score >= th else 0 for score in composite]
            metrics = EvaluationMetrics.calculate_classification_metrics(y_true, y_pred)
            metrics["threshold"] = round(th, 3)
            metrics["weights"] = {"schema": round(ws_n, 2), "reasoning": round(wr_n, 2), "nli": round(wc_n, 2)}
            results[cfg_name] = metrics

        return results
