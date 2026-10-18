"""Regression tests for optional dependency fallbacks."""

from app.core.ml_compat import safe_train_test_split
from app.interceptor.signals.reasoning_monitor import ReasoningMonitor


def test_reasoning_monitor_fallback_similarity():
    monitor = ReasoningMonitor()
    score = monitor._compute_semantic_similarity(
        "Canberra is the capital city of Australia.",
        ["Canberra is the capital city of Australia.", "Sydney is a major Australian city."],
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_train_test_split_fallback():
    traces = [{"id": idx} for idx in range(10)]
    train, test = safe_train_test_split(traces, train_size=0.5, random_state=42)
    assert len(train) + len(test) == len(traces)
    assert 0 < len(train) < len(traces)
