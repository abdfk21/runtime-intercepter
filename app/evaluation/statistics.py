"""Research statistical analysis: McNemar's test, Chi-square, Cohen's d, and hypothesis tests."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar
from app.evaluation.confidence_intervals import BootstrapCI


class StatisticalAnalyzer:
    """Rigorous statistical testing comparing experimental conditions and testing hypotheses."""

    @staticmethod
    def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size for two continuous samples."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        s1 = np.var(group1, ddof=1)
        s2 = np.var(group2, ddof=1)
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        if s_pooled == 0:
            return 0.0
        return float((np.mean(group1) - np.mean(group2)) / s_pooled)

    @classmethod
    def compare_on_off(
        cls,
        on_traces: List[Dict[str, Any]],
        off_traces: List[Dict[str, Any]],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Paired statistical comparison between INTERCEPTOR_ON and INTERCEPTOR_OFF."""
        # Align traces by task_id and fault_type
        off_dict = {
            f"{t['task_id']}_{t.get('fault_type')}": t
            for t in off_traces
            if t.get("fault_injected")
        }
        paired_on_contained: List[int] = []
        paired_off_contained: List[int] = []

        on_latencies: List[float] = [t.get("total_latency_ms", 0.0) for t in on_traces]
        off_latencies: List[float] = [t.get("total_latency_ms", 0.0) for t in off_traces]

        for t_on in on_traces:
            if not t_on.get("fault_injected"):
                continue
            key = f"{t_on['task_id']}_{t_on.get('fault_type')}"
            if key in off_dict:
                t_off = off_dict[key]
                paired_on_contained.append(1 if t_on.get("error_contained") else 0)
                paired_off_contained.append(1 if t_off.get("error_contained") else 0)

        n_pairs = len(paired_on_contained)
        if n_pairs == 0:
            return {
                "status": "No paired fault injection traces available for comparison.",
                "sample_size": 0,
            }

        # Build 2x2 contingency table for McNemar's test:
        # Table:
        #                OFF Contained (Yes)    OFF Contained (No)
        # ON Contained (Yes)       a                     b
        # ON Contained (No)        c                     d
        a = sum(1 for on, off in zip(paired_on_contained, paired_off_contained) if on == 1 and off == 1)
        b = sum(1 for on, off in zip(paired_on_contained, paired_off_contained) if on == 1 and off == 0)
        c = sum(1 for on, off in zip(paired_on_contained, paired_off_contained) if on == 0 and off == 1)
        d = sum(1 for on, off in zip(paired_on_contained, paired_off_contained) if on == 0 and off == 0)

        table = [[a, b], [c, d]]

        # McNemar test
        try:
            # Use exact binomial if discordant cells < 25
            exact = (b + c) < 25
            mcnemar_res = mcnemar(table, exact=exact)
            p_value = float(mcnemar_res.pvalue)
            statistic = float(mcnemar_res.statistic)
        except Exception:
            p_value = 1.0
            statistic = 0.0

        # Latency statistics & effect size
        ci_on_lat = BootstrapCI.compute_ci(on_latencies, alpha=alpha)
        ci_off_lat = BootstrapCI.compute_ci(off_latencies, alpha=alpha)
        cohen_d_latency = cls.calculate_cohens_d(on_latencies, off_latencies)

        on_containment_rate = sum(paired_on_contained) / n_pairs
        off_containment_rate = sum(paired_off_contained) / n_pairs

        return {
            "test_type": "McNemar's test for paired binary containment outcomes",
            "rationale": "Evaluates whether error containment changes significantly when interceptor is ON vs OFF across matched tasks.",
            "paired_sample_size": n_pairs,
            "contingency_table": table,
            "on_containment_rate": round(on_containment_rate, 4),
            "off_containment_rate": round(off_containment_rate, 4),
            "containment_difference": round(on_containment_rate - off_containment_rate, 4),
            "statistic": round(statistic, 4),
            "p_value": round(p_value, 5),
            "statistically_significant": bool(p_value < alpha),
            "alpha": alpha,
            "latency": {
                "on_latency": ci_on_lat,
                "off_latency": ci_off_lat,
                "cohen_d_overhead": round(cohen_d_latency, 3),
            },
        }

    @classmethod
    def test_hypotheses(
        cls,
        on_traces: List[Dict[str, Any]],
        off_traces: List[Dict[str, Any]],
        calibration_results: Dict[str, Any],
        ablation_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Formally evaluates empirical evidence for research hypotheses H1, H2, H3, H4."""
        # H1: Adaptive calibration reduces cross-model gap
        # Compare variance in F1 across models with fixed vs adaptive
        f1_fixed = calibration_results.get("fixed_f1_by_model", {})
        f1_adaptive = calibration_results.get("adaptive_f1_by_model", {})
        h1_supported = False
        h1_details = "Not evaluated"
        if f1_fixed and f1_adaptive:
            vals_fixed = list(f1_fixed.values())
            vals_adapt = list(f1_adaptive.values())
            range_fixed = max(vals_fixed) - min(vals_fixed)
            range_adapt = max(vals_adapt) - min(vals_adapt)
            h1_supported = range_adapt < range_fixed
            h1_details = f"Cross-model F1 range: Fixed={range_fixed:.3f}, Adaptive={range_adapt:.3f}"

        # H2: Runtime interception reduces cascade propagation compared to OFF
        comp = cls.compare_on_off(on_traces, off_traces)
        h2_supported = (
            comp.get("statistically_significant", False)
            and comp.get("on_containment_rate", 0.0) > comp.get("off_containment_rate", 0.0)
        )
        h2_details = (
            f"Containment Rate: ON={comp.get('on_containment_rate', 0.0):.1%} vs "
            f"OFF={comp.get('off_containment_rate', 0.0):.1%} (p={comp.get('p_value', 1.0):.4f})"
        )

        # H3: Cascade behavior differs across architectures
        arch_rates = {}
        for arch in ["sequential", "hierarchical", "parallel_consensus"]:
            arch_traces = [t for t in off_traces if t.get("architecture") == arch and t.get("fault_injected")]
            if arch_traces:
                casc = sum(1 for t in arch_traces if t.get("cascade_occurred"))
                arch_rates[arch] = casc / len(arch_traces)
        h3_supported = len(arch_rates) >= 2 and (max(arch_rates.values()) - min(arch_rates.values()) > 0.05)
        h3_details = f"Unchecked cascade rates by architecture: {arch_rates}"

        # H4: Full multi-signal fusion outperforms individual signals
        full_f1 = ablation_results.get("full_system", {}).get("f1", 0.0)
        schema_f1 = ablation_results.get("schema_only", {}).get("f1", 0.0)
        reason_f1 = ablation_results.get("reasoning_only", {}).get("f1", 0.0)
        nli_f1 = ablation_results.get("nli_only", {}).get("f1", 0.0)
        h4_supported = full_f1 >= max(schema_f1, reason_f1, nli_f1)
        h4_details = f"Full System F1: {full_f1:.3f} vs (Schema={schema_f1:.3f}, Reasoning={reason_f1:.3f}, NLI={nli_f1:.3f})"

        return {
            "H1_adaptive_calibration": {
                "hypothesis": "Adaptive per-model calibration reduces the cross-model effectiveness gap compared with a fixed global threshold.",
                "supported": h1_supported,
                "evidence": h1_details,
            },
            "H2_cascade_reduction": {
                "hypothesis": "Runtime interception reduces hallucination cascade propagation compared with interceptor OFF.",
                "supported": h2_supported,
                "evidence": h2_details,
            },
            "H3_architecture_differences": {
                "hypothesis": "Cascade behavior differs between sequential, hierarchical and parallel-consensus architectures.",
                "supported": h3_supported,
                "evidence": h3_details,
            },
            "H4_signal_fusion_synergy": {
                "hypothesis": "Combining schema validation, reasoning monitoring and NLI consistency performs differently from individual signals alone.",
                "supported": h4_supported,
                "evidence": h4_details,
            },
        }
