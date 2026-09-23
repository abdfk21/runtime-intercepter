"""End-to-End Test: Task -> Agent -> Interceptor -> Agent -> Interceptor -> Final Answer -> Evaluation."""

import pytest
from app.experiments.manager import ExperimentManager
from app.datasets.generator import DatasetGenerator
from app.evaluation.runner import EvaluationRunner


def test_full_pipeline_end_to_end(tmp_path):
    # 1. Run experiment on 4 tasks across 3 architectures with ON vs OFF
    mgr = ExperimentManager(seed=42)
    summary = mgr.run_experiment(
        models=["mock"],
        architectures=["sequential", "hierarchical", "parallel_consensus"],
        conditions=["INTERCEPTOR_ON", "INTERCEPTOR_OFF"],
        num_tasks=4,
        use_adaptive_calibration=False,
        save_to_db=True,
    )

    assert summary["total_traces_executed"] > 0
    assert summary["interceptor_on_traces"] > 0
    assert summary["interceptor_off_traces"] > 0
    assert "on_metrics" in summary
    assert "off_metrics" in summary

    # 2. Run evaluation runner to generate figures, tables, and manuscript summary
    runner = EvaluationRunner(output_dir=tmp_path)
    gen = DatasetGenerator(seed=42)
    tasks = gen.generate_tasks(count=4)

    artifacts = runner.generate_full_evaluation_artifacts(
        experiment_summary=summary,
        calibration_profiles=[],
        tasks=tasks,
    )

    assert "stat_results" in artifacts
    assert "ablation_results" in artifacts
    assert "report_path" in artifacts

    # Verify report file exists and has content
    report_file = tmp_path / "research_summary.md"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Research Summary" in content
    assert "Error Containment Rate" in content
