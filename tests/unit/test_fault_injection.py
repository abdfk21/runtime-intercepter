"""Unit tests for controlled fault injection."""

import pytest
from app.fault_injection.injector import FaultInjector
from app.fault_injection.fault_types import FaultType


@pytest.mark.parametrize("fault_type", list(FaultType))
def test_all_ten_fault_types(sample_valid_output, fault_type):
    injector = FaultInjector(seed=42)
    res = injector.inject_fault(sample_valid_output, fault_type=fault_type, seed=42)

    assert res.fault_id.startswith("flt_")
    assert res.fault_type == fault_type
    assert res.corrupted_output is not None
    assert res.ground_truth is not None
    assert res.random_seed == 42


def test_fault_injection_reproducibility(sample_valid_output):
    injector = FaultInjector(seed=999)
    res1 = injector.inject_fault(sample_valid_output, fault_type=FaultType.FACTUAL_MUTATION, seed=999)
    res2 = injector.inject_fault(sample_valid_output, fault_type=FaultType.FACTUAL_MUTATION, seed=999)

    assert res1.corrupted_output == res2.corrupted_output
