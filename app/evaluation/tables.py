"""Generates academic tables in CSV, Markdown, and LaTeX formats."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class ResearchTableGenerator:
    """Generates standardized academic research tables (Tables 1-9) across CSV, Markdown, and LaTeX."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results/tables")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _export(self, df: pd.DataFrame, table_id: str, title: str):
        # CSV
        df.to_csv(self.output_dir / f"{table_id}.csv", index=False)
        # Markdown
        with open(self.output_dir / f"{table_id}.md", "w", encoding="utf-8") as f:
            f.write(f"### {title}\n\n")
            try:
                f.write(df.to_markdown(index=False))
            except Exception:
                headers = list(df.columns)
                f.write("| " + " | ".join(str(h) for h in headers) + " |\n")
                f.write("| " + " | ".join("---" for _ in headers) + " |\n")
                for _, row in df.iterrows():
                    f.write("| " + " | ".join(str(v) for v in row) + " |\n")
            f.write("\n")
        # LaTeX
        with open(self.output_dir / f"{table_id}.tex", "w", encoding="utf-8") as f:
            f.write(f"% {title}\n")
            try:
                f.write(df.to_latex(index=False, caption=title, label=f"tab:{table_id}"))
            except Exception:
                f.write("% LaTeX generation fallback\n")

    def generate_table1_dataset_distribution(self, tasks: List[Any]):
        cat_counts = {}
        for t in tasks:
            cat = getattr(t, "category", None) or t.get("category")
            val = cat.value if hasattr(cat, "value") else str(cat)
            cat_counts[val] = cat_counts.get(val, 0) + 1

        data = [{"Task Category": k.replace("_", " ").title(), "Count": v, "Percentage (%)": round(v / max(1, len(tasks)) * 100, 1)} for k, v in cat_counts.items()]
        df = pd.DataFrame(data)
        self._export(df, "table1_dataset_distribution", "Table 1: Benchmark Dataset Task Distribution")

    def generate_table2_model_configuration(self, models_cfg: Dict[str, Any]):
        rows = []
        for m_name, m_info in models_cfg.items():
            rows.append({
                "Model Identifier": m_name,
                "Provider": m_info.get("provider", "N/A"),
                "Active Model": m_info.get("model_name", "N/A"),
                "Temperature": m_info.get("temperature", 0.0),
                "Max Tokens": m_info.get("max_tokens", 1024),
            })
        df = pd.DataFrame(rows)
        self._export(df, "table2_model_configuration", "Table 2: Evaluated LLM Models and Runtime Parameters")

    def generate_table3_detection_performance(self, metrics: Dict[str, float]):
        rows = [{
            "Accuracy": f"{metrics.get('accuracy', 0):.3f}",
            "Precision": f"{metrics.get('precision', 0):.3f}",
            "Recall": f"{metrics.get('recall', 0):.3f}",
            "Specificity": f"{metrics.get('specificity', 0):.3f}",
            "F1-Score": f"{metrics.get('f1', 0):.3f}",
            "FPR": f"{metrics.get('fpr', 0):.3f}",
            "FNR": f"{metrics.get('fnr', 0):.3f}",
        }]
        df = pd.DataFrame(rows)
        self._export(df, "table3_detection_performance", "Table 3: Multi-Signal Anomaly Detection Performance")

    def generate_table4_interceptor_on_off(self, on_metrics: Dict[str, Any], off_metrics: Dict[str, Any]):
        rows = [
            {"Metric": "Cascade Occurrence Rate (%)", "Interceptor OFF": f"{off_metrics.get('cascade_rate', 0)*100:.1f}%", "Interceptor ON": f"{on_metrics.get('cascade_rate', 0)*100:.1f}%"},
            {"Metric": "Error Containment Rate (%)", "Interceptor OFF": f"{off_metrics.get('error_containment_rate', 0)*100:.1f}%", "Interceptor ON": f"{on_metrics.get('error_containment_rate', 0)*100:.1f}%"},
            {"Metric": "Mean Cascade Depth", "Interceptor OFF": f"{off_metrics.get('mean_cascade_depth', 0):.2f}", "Interceptor ON": f"{on_metrics.get('mean_cascade_depth', 0):.2f}"},
            {"Metric": "Mean Cascade Severity", "Interceptor OFF": f"{off_metrics.get('mean_cascade_severity', 0):.2f}", "Interceptor ON": f"{on_metrics.get('mean_cascade_severity', 0):.2f}"},
            {"Metric": "Avg Pipeline Latency (ms)", "Interceptor OFF": f"{off_metrics.get('avg_pipeline_latency_ms', 0):.1f}", "Interceptor ON": f"{on_metrics.get('avg_pipeline_latency_ms', 0):.1f}"},
        ]
        df = pd.DataFrame(rows)
        self._export(df, "table4_interceptor_on_off", "Table 4: Comparative Pipeline Performance Under ON vs. OFF Conditions")

    def generate_table5_architecture_comparison(self, arch_results: Dict[str, Dict[str, float]]):
        rows = []
        for arch, vals in arch_results.items():
            rows.append({
                "Architecture": arch.replace("_", " ").title(),
                "Unchecked Cascade Rate (%)": f"{vals.get('off_cascade', 0)*100:.1f}%",
                "Intercepted Containment (%)": f"{vals.get('on_containment', 0)*100:.1f}%",
                "Mean Cascade Depth (OFF)": f"{vals.get('mean_depth_off', 0):.2f}",
                "Intervention Rate (%)": f"{vals.get('intervention_rate', 0)*100:.1f}%",
            })
        df = pd.DataFrame(rows)
        self._export(df, "table5_architecture_comparison", "Table 5: Cascade Dynamics Across Multi-Agent Orchestration Topologies")

    def generate_table6_calibration_results(self, profiles: List[Any]):
        rows = []
        for p in profiles:
            rows.append({
                "Model": getattr(p, "model_name", str(p)),
                "Calibrated Threshold": f"{getattr(p, 'threshold', 0.5):.3f}",
                "Precision": f"{getattr(p, 'precision', 0):.3f}",
                "Recall": f"{getattr(p, 'recall', 0):.3f}",
                "F1 Score": f"{getattr(p, 'f1', 0):.3f}",
                "FPR": f"{getattr(p, 'false_positive_rate', 0):.3f}",
                "Val Size": getattr(p, "validation_size", 0),
            })
        df = pd.DataFrame(rows)
        self._export(df, "table6_calibration_results", "Table 6: Empirical Per-Model Adaptive Calibration Profiles")

    def generate_table7_ablation_results(self, ablation_dict: Dict[str, Dict[str, Any]]):
        rows = []
        for cfg, vals in ablation_dict.items():
            rows.append({
                "Configuration": cfg.replace("_", " ").title(),
                "Threshold": f"{vals.get('threshold', 0.5):.2f}",
                "Precision": f"{vals.get('precision', 0):.3f}",
                "Recall": f"{vals.get('recall', 0):.3f}",
                "F1 Score": f"{vals.get('f1', 0):.3f}",
                "Accuracy": f"{vals.get('accuracy', 0):.3f}",
            })
        df = pd.DataFrame(rows)
        self._export(df, "table7_ablation_results", "Table 7: Ablation Study Across Individual Signals and Calibration")

    def generate_table8_inter_annotator_agreement(self, agreement_data: Dict[str, Any]):
        rows = [{
            "Raters": f"{agreement_data.get('annotator_a', 'R1')} & {agreement_data.get('annotator_b', 'R2')}",
            "Overlap Count": agreement_data.get("common_count", 0),
            "Disagreements": agreement_data.get("disagreements", 0),
            "Raw Agreement (%)": f"{agreement_data.get('agreement_rate', 0)*100:.1f}%",
            "Cohen's Kappa": f"{agreement_data.get('cohen_kappa', 0):.3f}" if agreement_data.get('cohen_kappa') is not None else "N/A",
            "Status": agreement_data.get("status", "Evaluated"),
        }]
        df = pd.DataFrame(rows)
        self._export(df, "table8_inter_annotator_agreement", "Table 8: Inter-Annotator Agreement Statistics")

    def generate_table9_latency_overhead(self, latency_dict: Dict[str, float]):
        rows = []
        for stage, ms in latency_dict.items():
            rows.append({
                "Stage Component": stage.replace("_", " ").title(),
                "Latency (ms)": f"{ms:.2f}",
            })
        df = pd.DataFrame(rows)
        self._export(df, "table9_latency_overhead", "Table 9: Runtime Interception Latency Overhead Breakdown")
