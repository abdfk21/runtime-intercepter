"""Publication-quality visualization generator saving in PNG and SVG formats."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Generates research figures for multi-agent hallucination interception."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("results/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Publication styling
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.autolayout": True,
        })

    def _save_fig(self, fig: plt.Figure, base_name: str):
        png_path = self.output_dir / f"{base_name}.png"
        svg_path = self.output_dir / f"{base_name}.svg"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)

    def plot_cascade_rate_on_vs_off(self, on_rate: float, off_rate: float):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(["Interceptor OFF", "Interceptor ON"], [off_rate * 100, on_rate * 100], color=["#d9534f", "#5cb85c"], width=0.5)
        ax.set_ylabel("Cascade Occurrence Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Hallucination Cascade Rate: Interceptor OFF vs. ON")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        self._save_fig(fig, "fig1_cascade_rate_on_vs_off")

    def plot_error_containment(self, on_contain: float, off_contain: float):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        bars = ax.bar(["Interceptor OFF", "Interceptor ON"], [off_contain * 100, on_contain * 100], color=["#f0ad4e", "#0275d8"], width=0.5)
        ax.set_ylabel("Error Containment Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Error Containment Rate: Interceptor OFF vs. ON")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
        self._save_fig(fig, "fig2_error_containment")

    def plot_architecture_comparison(self, arch_metrics: Dict[str, Dict[str, float]]):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        archs = list(arch_metrics.keys())
        x = np.arange(len(archs))
        width = 0.35

        on_rates = [arch_metrics[a].get("on_containment", 0.0) * 100 for a in archs]
        off_rates = [arch_metrics[a].get("off_containment", 0.0) * 100 for a in archs]

        ax.bar(x - width/2, off_rates, width, label="OFF (Unchecked)", color="#d9534f")
        ax.bar(x + width/2, on_rates, width, label="ON (Intercepted)", color="#5cb85c")

        ax.set_ylabel("Error Containment Rate (%)")
        ax.set_title("Error Containment by Multi-Agent Architecture")
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", " ").title() for a in archs])
        ax.set_ylim(0, 110)
        ax.legend(loc="upper left")
        self._save_fig(fig, "fig3_architecture_comparison")

    def plot_ablation_results(self, ablation_data: Dict[str, Dict[str, float]]):
        fig, ax = plt.subplots(figsize=(10, 5))
        configs = list(ablation_data.keys())
        f1_scores = [ablation_data[c].get("f1", 0.0) * 100 for c in configs]
        clean_names = [c.replace("_", " ").title() for c in configs]

        colors = ["#0275d8" if "Full" in name or "Adaptive" in name else "#6c757d" for name in clean_names]
        bars = ax.barh(clean_names, f1_scores, color=colors)
        ax.set_xlabel("F1 Detection Score (%)")
        ax.set_title("Ablation Study: Signal Contribution and Calibration Impact")
        ax.set_xlim(0, 105)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height()/2.0, f"{w:.1f}%", ha='left', va='center')
        self._save_fig(fig, "fig4_ablation_study")

    def plot_calibration_curves(self, search_history: List[Dict[str, float]], optimal_th: float):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        thresholds = [m["threshold"] for m in search_history]
        f1s = [m["f1"] for m in search_history]
        precs = [m["precision"] for m in search_history]
        recs = [m["recall"] for m in search_history]

        ax.plot(thresholds, f1s, label="F1 Score", color="#0275d8", linewidth=2.2)
        ax.plot(thresholds, precs, label="Precision", color="#5cb85c", linestyle="--", linewidth=1.8)
        ax.plot(thresholds, recs, label="Recall", color="#d9534f", linestyle=":", linewidth=1.8)

        ax.axvline(optimal_th, color="black", linestyle="-.", label=f"Optimal ({optimal_th:.2f})")
        ax.set_xlabel("Decision Threshold (Risk Score)")
        ax.set_ylabel("Score")
        ax.set_title("Threshold Optimization Curve")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower center")
        self._save_fig(fig, "fig5_calibration_curve")

    def plot_latency_overhead(self, latency_dict: Dict[str, float]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        categories = list(latency_dict.keys())
        vals = list(latency_dict.values())
        clean_cats = [c.replace("_", " ").title() for c in categories]

        bars = ax.bar(clean_cats, vals, color="#5bc0de", width=0.5)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Runtime Interceptor Latency Overhead Breakdown")
        for bar in bars:
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, y + max(vals)*0.02, f"{y:.1f} ms", ha='center', va='bottom')
        self._save_fig(fig, "fig6_latency_overhead")
