"""Unit tests for Evaluation Metrics, Confidence Intervals, and Statistics."""

import pytest
from app.evaluation.metrics import EvaluationMetrics
from app.evaluation.confidence_intervals import BootstrapCI
from app.evaluation.statistics import StatisticalAnalyzer
from app.evaluation.agreement import AgreementCalculator
import numpy as np


def test_classification_metrics():
    y_true = [1, 1, 0, 0, 1]
    y_pred = [1, 0, 0, 1, 1]
    m = EvaluationMetrics.calculate_classification_metrics(y_true, y_pred)

    assert m["tp"] == 2
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["tn"] == 1
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0


def test_cascade_metrics():
    traces = [
        {"fault_injected": True, "cascade_occurred": False, "error_contained": True, "total_latency_ms": 100},
        {"fault_injected": True, "cascade_occurred": True, "error_contained": False, "total_latency_ms": 120},
    ]
    m = EvaluationMetrics.calculate_cascade_metrics(traces)
    assert m["cascade_rate"] == 0.5
    assert m["error_containment_rate"] == 0.5


def test_bootstrap_ci():
    data = [10.0, 12.0, 11.5, 9.8, 10.5, 11.2]
    ci = BootstrapCI.compute_ci(data, iterations=500, alpha=0.05)
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]


def test_cohens_d():
    g1 = [10.0, 11.0, 12.0, 10.5]
    g2 = [14.0, 15.0, 14.5, 16.0]
    d = StatisticalAnalyzer.calculate_cohens_d(g1, g2)
    assert d < 0  # g1 is smaller than g2


def test_fleiss_kappa():
    # 5 subjects, 3 categories, 3 raters
    matrix = np.array([
        [3, 0, 0],
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
        [0, 0, 3],
    ])
    kappa = AgreementCalculator.calculate_fleiss_kappa(matrix)
    assert kappa == 1.0  # Perfect agreement
