"""Evaluation metrics for detection performance and multi-agent cascade containment.

Formulas:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall (Sensitivity) = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
- False Positive Rate (FPR) = FP / (FP + TN)
- False Negative Rate (FNR) = FN / (FN + TP)
- Detection Rate = TP / Total Faults
- Cascade Rate = Faults Propagated to Final Output / Total Injected Faults
- Cascade Depth = Number of downstream agents affected by error
- Error Containment Rate = Prevented Faults / Total Injected Faults = (1 - Cascade Rate)
- Repair Success Rate = Successful Repairs / Total Repair Invocations
- Intervention Rate = Interventions (Warning + Repair + Block) / Total Handoffs
"""

from typing import Any, Dict, List
import numpy as np


class EvaluationMetrics:
    """Calculates research metrics for detection and error containment."""

    @staticmethod
    def calculate_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        y_t = np.array(y_true, dtype=int)
        y_p = np.array(y_pred, dtype=int)

        tp = int(np.sum((y_p == 1) & (y_t == 1)))
        fp = int(np.sum((y_p == 1) & (y_t == 0)))
        tn = int(np.sum((y_p == 0) & (y_t == 0)))
        fn = int(np.sum((y_p == 0) & (y_t == 1)))

        total = max(1, len(y_t))
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        detection_rate = recall

        return {
            "sample_size": total,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "specificity": round(float(specificity), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "fnr": round(float(fnr), 4),
            "detection_rate": round(float(detection_rate), 4),
        }

    @staticmethod
    def calculate_cascade_metrics(traces: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculates multi-agent cascade metrics: cascade rate, depth, containment rate, latencies."""
        total_traces = len(traces)
        if total_traces == 0:
            return {}

        fault_traces = [t for t in traces if t.get("fault_injected", False)]
        total_faults = max(1, len(fault_traces))

        cascades = [t for t in fault_traces if t.get("cascade_occurred", False)]
        contained = [t for t in fault_traces if t.get("error_contained", False)]

        cascade_depths = [t.get("cascade_depth", 0) for t in fault_traces]
        cascade_severities = [t.get("cascade_severity", 0.0) for t in fault_traces]

        total_latencies = [t.get("total_latency_ms", 0.0) for t in traces]
        interceptor_latencies = [t.get("interceptor_latency_ms", 0.0) for t in traces]
        repair_latencies = [t.get("repair_latency_ms", 0.0) for t in traces]

        cascade_rate = len(cascades) / total_faults
        containment_rate = len(contained) / total_faults

        # Handoff intervention breakdown
        total_handoffs = 0
        warning_handoffs = 0
        repair_handoffs = 0
        block_handoffs = 0

        for t in traces:
            for h in t.get("handoffs", []):
                total_handoffs += 1
                dec = h.get("decision")
                if dec == "PASS_WITH_WARNING":
                    warning_handoffs += 1
                elif dec == "REPAIR":
                    repair_handoffs += 1
                elif dec == "BLOCK":
                    block_handoffs += 1

        total_h_safe = max(1, total_handoffs)

        return {
            "total_traces": total_traces,
            "total_faults_injected": len(fault_traces),
            "cascade_count": len(cascades),
            "cascade_rate": round(float(cascade_rate), 4),
            "error_containment_rate": round(float(containment_rate), 4),
            "mean_cascade_depth": round(float(np.mean(cascade_depths)) if cascade_depths else 0.0, 3),
            "mean_cascade_severity": round(float(np.mean(cascade_severities)) if cascade_severities else 0.0, 3),
            "total_handoffs": total_handoffs,
            "warning_rate": round(warning_handoffs / total_h_safe, 4),
            "repair_rate": round(repair_handoffs / total_h_safe, 4),
            "block_rate": round(block_handoffs / total_h_safe, 4),
            "intervention_rate": round((warning_handoffs + repair_handoffs + block_handoffs) / total_h_safe, 4),
            "avg_pipeline_latency_ms": round(float(np.mean(total_latencies)) if total_latencies else 0.0, 2),
            "avg_interceptor_latency_ms": round(float(np.mean(interceptor_latencies)) if interceptor_latencies else 0.0, 2),
            "avg_repair_latency_ms": round(float(np.mean(repair_latencies)) if repair_latencies else 0.0, 2),
        }
