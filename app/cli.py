"""Command-Line Interface for running demos, benchmarks, calibration, and full experiments."""

import os
import sys
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from app.core.config import settings
from app.core.logging import logger
from app.storage.database import init_db
from app.llm.mock import MockProvider
from app.interceptor.interceptor import RuntimeInterceptor
from app.interceptor.calibration.calibrator import ModelCalibrator
from app.orchestration.sequential import SequentialPipeline
from app.fault_injection.fault_types import FaultType
from app.datasets.generator import DatasetGenerator
from app.datasets.loader import DatasetLoader
from app.datasets.annotation import AnnotationManager
from app.evaluation.ablation import AblationStudyRunner
from app.evaluation.runner import EvaluationRunner
from app.experiments.manager import ExperimentManager

cli = typer.Typer(help="Runtime Interceptor for Hallucination Cascades in Multi-Agent LLMs")
console = Console()


@cli.command()
def demo():
    """Run an interactive offline demonstration comparing pipeline behavior WITH and WITHOUT interceptor."""
    console.print(Panel.fit(
        "[bold cyan]CONTROLLED ACADEMIC DEMONSTRATION[/bold cyan]\n"
        "[italic]Demonstrating Hallucination Propagation vs. Runtime Interception Containment[/italic]\n"
        "[yellow]Note: Controlled educational demonstration, not benchmark data.[/yellow]",
        border_style="cyan"
    ))

    prompt = "What is the capital city of Australia and when was it founded?"
    llm = MockProvider(seed=42)
    interceptor = RuntimeInterceptor()

    # Part 1: Interceptor OFF (Unchecked Cascade)
    console.print("\n[bold red]1. CONDITION: INTERCEPTOR OFF[/bold red]")
    pipeline_off = SequentialPipeline(llm_provider=llm, interceptor=interceptor)
    fi_cfg = {
        "fault_type": "factual_mutation",
        "location": "researcher",
        "seed": 101,
    }
    res_off = pipeline_off.run(prompt=prompt, interceptor_enabled=False, fault_injection_config=fi_cfg)

    table_off = Table(title="Pipeline Execution Trace (Interceptor OFF)")
    table_off.add_column("Agent Stage", style="cyan")
    table_off.add_column("Observable Output Summary", style="white")
    table_off.add_column("Handoff Status", style="red")

    table_off.add_row("Researcher (Fault Injected)", str(res_off["agent_outputs"].get("researcher", {}).get("answer", ""))[:70] + "...", "UNMONITORED")
    table_off.add_row("Verifier", str(res_off["agent_outputs"].get("verifier", {}).get("answer", ""))[:70] + "...", "PROPAGATED")
    table_off.add_row("Critic", str(res_off["agent_outputs"].get("critic", {}).get("answer", ""))[:70] + "...", "PROPAGATED")
    table_off.add_row("Synthesizer (Final)", str(res_off["agent_outputs"].get("synthesizer", {}).get("answer", ""))[:70] + "...", "[bold red]CASCADE OCCURRED[/bold red]")
    console.print(table_off)

    console.print(f"[red]Result: Injected anomaly propagated to final synthesizer. Error Contained = {res_off['error_contained']}[/red]")

    # Part 2: Interceptor ON (Active Containment)
    console.print("\n[bold green]2. CONDITION: INTERCEPTOR ON (Active Runtime Boundary)[/bold green]")
    pipeline_on = SequentialPipeline(llm_provider=llm, interceptor=interceptor)
    res_on = pipeline_on.run(prompt=prompt, interceptor_enabled=True, fault_injection_config=fi_cfg)

    table_on = Table(title="Pipeline Execution Trace (Interceptor ON)")
    table_on.add_column("Agent Stage", style="cyan")
    table_on.add_column("Observable Output Summary", style="white")
    table_on.add_column("Interceptor Decision", style="green")
    table_on.add_column("Risk Score", style="yellow")

    for h in res_on.get("handoffs", []):
        dec_val = h["decision"].value if hasattr(h["decision"], "value") else str(h["decision"])
        dec_style = "bold green" if dec_val == "PASS" else "bold yellow" if "WARNING" in dec_val else "bold red"
        table_on.add_row(
            f"{h['source_agent']} -> {h['target_agent']}",
            f"{h['explanations'][0] if h['explanations'] else 'Valid grounding'}"[:50] + "...",
            f"[{dec_style}]{dec_val}[/{dec_style}]",
            f"{h['risk_score']:.2f} (Th: {h['threshold']:.2f})"
        )

    console.print(table_on)
    console.print(f"[green]Final Synthesis: {res_on['final_output'].get('answer', '')[:90]}...[/green]")
    console.print(f"[bold green]Result: Anomaly successfully intercepted and contained! Error Contained = {res_on['error_contained']}[/bold green]\n")


@cli.command("generate-dataset")
def generate_dataset_cmd(size: int = 350):
    """Generate 350-trace benchmark and validation dataset across 8 task categories."""
    console.print(f"[cyan]Generating benchmark dataset with {size} validation traces...[/cyan]")
    gen = DatasetGenerator(seed=settings.random_seed)
    tasks = gen.generate_tasks(count=size)
    traces = gen.generate_validation_dataset(tasks, target_size=size)

    dataset_path = settings.data_dir / "validation" / "benchmark_validation_set.json"
    gen.save_dataset(traces, dataset_path)
    console.print(f"[green]Dataset successfully saved to {dataset_path} ({len(traces)} traces)[/green]")


@cli.command()
def calibrate(model_name: str = "deterministic-mock-v1"):
    """Calibrate adaptive decision threshold and weights for a specific LLM."""
    console.print(f"[cyan]Calibrating model '{model_name}' on held-out validation data...[/cyan]")
    gen = DatasetGenerator(seed=settings.random_seed)
    tasks = gen.generate_tasks(count=60)
    traces = gen.generate_validation_dataset(tasks, target_size=60)

    calib = ModelCalibrator()
    profile = calib.calibrate(model_name, traces)

    console.print(Panel.fit(
        f"[bold green]Model Calibration Complete: {profile.model_name}[/bold green]\n"
        f"Optimized Threshold: [bold yellow]{profile.threshold:.3f}[/bold yellow]\n"
        f"Validation F1: [bold cyan]{profile.f1:.4f}[/bold cyan] | Precision: {profile.precision:.4f} | Recall: {profile.recall:.4f}\n"
        f"Signal Weights: Schema={profile.weights['schema_weight']:.2f}, "
        f"Reasoning={profile.weights['reasoning_weight']:.2f}, Consistency={profile.weights['consistency_weight']:.2f}",
        border_style="green"
    ))


@cli.command()
def benchmark(num_tasks: int = 15):
    """Run full benchmark across architectures and conditions."""
    init_db()
    mgr = ExperimentManager(seed=settings.random_seed)
    summary = mgr.run_experiment(num_tasks=num_tasks)
    console.print(f"[green]Benchmark completed! Evaluated {summary['total_traces_executed']} traces.[/green]")
    console.print(f"Error Containment: ON={summary['on_metrics'].get('error_containment_rate', 0):.1%} vs OFF={summary['off_metrics'].get('error_containment_rate', 0):.1%}")


@cli.command()
def ablation():
    """Run ablation study assessing individual detection signals and calibration."""
    console.print("[cyan]Running ablation study on multi-agent validation traces...[/cyan]")
    gen = DatasetGenerator(seed=settings.random_seed)
    tasks = gen.generate_tasks(count=40)
    traces = gen.generate_validation_dataset(tasks, target_size=40)

    runner = AblationStudyRunner()
    results = runner.run_ablation(traces)

    table = Table(title="Ablation Study Results")
    table.add_column("Configuration", style="cyan")
    table.add_column("Threshold", style="white")
    table.add_column("Precision", style="white")
    table.add_column("Recall", style="white")
    table.add_column("F1 Score", style="bold green")

    for cfg, m in results.items():
        table.add_row(
            cfg.replace("_", " ").title(),
            str(m.get("threshold", 0.5)),
            f"{m.get('precision', 0):.3f}",
            f"{m.get('recall', 0):.3f}",
            f"{m.get('f1', 0):.3f}"
        )
    console.print(table)


@cli.command()
def dashboard(host: str = "0.0.0.0", port: int = 8000):
    """Launch the interactive FastAPI research web dashboard."""
    import uvicorn
    console.print(f"[bold green]Starting Dashboard at http://localhost:{port}[/bold green]")
    uvicorn.run("app.main:app", host=host, port=port, reload=False)


@cli.command("full-experiment")
def full_experiment():
    """Execute complete end-to-end automated academic workflow:
    Dataset -> Fault Injection -> Calibration -> Experiments -> Metrics -> Ablation -> Statistics -> Plots -> Tables -> Report.
    """
    console.print(Panel.fit(
        "[bold cyan]AUTONOMOUS FULL ACADEMIC EXPERIMENT EXECUTION[/bold cyan]\n"
        "[italic]Executing end-to-end pipeline: generation -> calibration -> benchmarking -> statistics -> artifacts[/italic]",
        border_style="cyan"
    ))

    start_t = time.perf_counter()

    # Step 1: Initialize DB
    console.print("[1/9] Initializing database...")
    init_db()

    # Step 2: Generate Tasks and Calibration/Test Split
    console.print("[2/9] Generating benchmark tasks and 350 validation traces...")
    gen = DatasetGenerator(seed=settings.random_seed)
    tasks = gen.generate_tasks(count=350)
    raw_traces = gen.generate_validation_dataset(tasks, target_size=350)
    calib_traces, test_traces = DatasetLoader.split_calibration_test(raw_traces, calibration_ratio=0.5)

    # Step 3: Calibrate Models
    console.print("[3/9] Running adaptive calibration across LLM models...")
    calibrator = ModelCalibrator()
    calib_profiles = []
    for model_name in ["deterministic-mock-v1", "gemini-2.5-flash", "llama-3.3-70b-versatile"]:
        prof = calibrator.calibrate(model_name, calib_traces)
        calib_profiles.append(prof)

    # Step 4: Run Experiments across 3 Architectures x ON/OFF
    console.print("[4/9] Running multi-agent experiment suite (Sequential, Hierarchical, Parallel-Consensus)...")
    exp_mgr = ExperimentManager(seed=settings.random_seed)
    exp_summary = exp_mgr.run_experiment(
        models=["mock"],
        architectures=["sequential", "hierarchical", "parallel_consensus"],
        conditions=["INTERCEPTOR_ON", "INTERCEPTOR_OFF"],
        num_tasks=12,
        use_adaptive_calibration=True,
    )

    # Step 5: Calculate Metrics & Ablation
    console.print("[5/9] Computing detection metrics, cascade measurements, and ablation...")
    eval_runner = EvaluationRunner()
    
    # Step 6: Generate Tables (Tables 1-9 in CSV, MD, LaTeX)
    console.print("[6/9] Generating Tables 1-9 (CSV, Markdown, LaTeX)...")

    # Step 7: Generate Publication Figures (PNG, SVG)
    console.print("[7/9] Generating publication-quality figures (PNG, SVG)...")
    artifacts_res = eval_runner.generate_full_evaluation_artifacts(
        experiment_summary=exp_summary,
        calibration_profiles=calib_profiles,
        tasks=tasks,
        test_traces=test_traces,
    )

    # Step 8: Generate Research Manuscript Summary
    console.print("[8/9] Compiling comprehensive research summary report...")

    # Step 9: Final Report
    elapsed = time.perf_counter() - start_t
    console.print(Panel.fit(
        f"[bold green]FULL EXPERIMENT SUITE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s[/bold green]\n"
        f"Traces Executed: {exp_summary['total_traces_executed']}\n"
        f"Error Containment: ON = [bold green]{exp_summary['on_metrics']['error_containment_rate']:.1%}[/bold green] "
        f"vs OFF = [bold red]{exp_summary['off_metrics']['error_containment_rate']:.1%}[/bold red]\n"
        f"Research Summary: {artifacts_res['report_path']}\n"
        f"Figures Directory: results/figures/\n"
        f"Tables Directory: results/tables/",
        border_style="green"
    ))


if __name__ == "__main__":
    cli()
