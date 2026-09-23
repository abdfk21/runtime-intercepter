"""Unit tests for threshold optimization and model calibration."""

import pytest
from app.interceptor.calibration.threshold_optimizer import ThresholdOptimizer
from app.interceptor.calibration.calibrator import ModelCalibrator
from app.datasets.generator import DatasetGenerator


def test_threshold_optimizer_evaluation():
    y_true = [1, 1, 0, 0, 1]
    scores = [0.8, 0.7, 0.2, 0.3, 0.9]
    metrics = ThresholdOptimizer.evaluate_threshold(y_true, scores, threshold=0.5)

    assert metrics["tp"] == 3
    assert metrics["fp"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_threshold_optimizer_search():
    y_true = [1, 1, 0, 0, 1, 0, 1, 0]
    scores = [0.85, 0.72, 0.15, 0.30, 0.91, 0.40, 0.68, 0.22]
    opt = ThresholdOptimizer.optimize(y_true, scores, target_metric="f1", min_precision=0.6)

    assert "best_threshold" in opt
    assert opt["best_metrics"]["f1"] > 0.80


def test_model_calibrator_pipeline(tmp_path):
    calibrator = ModelCalibrator(profiles_dir=tmp_path)
    gen = DatasetGenerator(seed=42)
    tasks = gen.generate_tasks(count=20)
    traces = gen.generate_validation_dataset(tasks, target_size=20)

    profile = calibrator.calibrate(
        model_name="test-model-v1",
        calibration_traces=traces,
        optimization_target="f1",
    )

    assert profile.model_name == "test-model-v1"
    assert 0.1 <= profile.threshold <= 0.9
    assert profile.validation_size == 20

    # Verify load
    loaded = calibrator.load_profile("test-model-v1")
    assert loaded is not None
    assert loaded.threshold == profile.threshold
