"""Unit tests for Detection Signals 1, 2, 3 and Signal Fusion."""

import pytest
from app.interceptor.signals.schema_validator import SchemaValidator
from app.interceptor.signals.reasoning_monitor import ReasoningMonitor
from app.interceptor.signals.nli_consistency_checker import NLIConsistencyChecker
from app.interceptor.fusion import SignalFusion


def test_schema_validator_valid(sample_valid_output):
    validator = SchemaValidator()
    sig = validator.validate(sample_valid_output)
    assert sig.passed is True
    assert sig.normalized_score == 0.0
    assert sig.details["schema_error_count"] == 0


def test_schema_validator_invalid():
    validator = SchemaValidator()
    malformed = {"answer": "No claims key here"}
    sig = validator.validate(malformed)
    assert sig.passed is False
    assert sig.normalized_score > 0.0
    assert sig.details["schema_error_count"] > 0


def test_reasoning_monitor_grounded(sample_valid_output):
    monitor = ReasoningMonitor()
    sig = monitor.evaluate(sample_valid_output)
    assert sig.passed is True
    assert sig.normalized_score < 0.35


def test_reasoning_monitor_unsupported(sample_corrupted_output):
    monitor = ReasoningMonitor()
    sig = monitor.evaluate(sample_corrupted_output)
    assert sig.normalized_score > 0.30
    assert sig.details["unsupported_claims"] > 0 or sig.details["numerical_mismatch_count"] > 0


def test_nli_consistency_entailment():
    nli = NLIConsistencyChecker()
    label, p_ent, p_neu, p_con = nli.classify_pair(
        premise="Canberra is the federal capital of Australia.",
        hypothesis="Canberra is the capital city of Australia."
    )
    assert label == "ENTAILMENT"
    assert p_ent > 0.50


def test_nli_consistency_contradiction():
    nli = NLIConsistencyChecker()
    label, p_ent, p_neu, p_con = nli.classify_pair(
        premise="The capital of Australia is Canberra.",
        hypothesis="The capital of Australia is Sydney."
    )
    assert label == "CONTRADICTION"
    assert p_con > 0.40


def test_signal_fusion():
    fusion = SignalFusion(schema_weight=0.25, reasoning_weight=0.40, consistency_weight=0.35)
    validator = SchemaValidator()
    monitor = ReasoningMonitor()
    nli = NLIConsistencyChecker()

    out = {
        "answer": "Test answer",
        "claims": [{"claim": "Test claim", "confidence": 0.9, "evidence": ["Test evidence"]}],
        "assumptions": [],
        "sources": []
    }
    s1 = validator.validate(out)
    s2 = monitor.evaluate(out)
    s3 = nli.evaluate(out, {"premise": "Test premise"})

    risk, weights = fusion.fuse(s1, s2, s3)
    assert 0.0 <= risk <= 1.0
    assert abs(sum(weights.values()) - 1.0) < 1e-4
