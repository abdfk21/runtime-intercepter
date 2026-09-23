"""Integration tests for Sequential, Hierarchical, and Parallel-Consensus architectures."""

import pytest
from app.orchestration.sequential import SequentialPipeline
from app.orchestration.hierarchical import HierarchicalPipeline
from app.orchestration.parallel_consensus import ParallelConsensusPipeline


def test_sequential_pipeline_clean(mock_llm, interceptor, fault_injector):
    pipeline = SequentialPipeline(llm_provider=mock_llm, interceptor=interceptor, fault_injector=fault_injector)
    res = pipeline.run("What is the capital of Australia?", interceptor_enabled=True)

    assert res["final_output"] is not None
    assert "answer" in res["final_output"]
    assert res["error_contained"] is True
    assert res["total_latency_ms"] > 0


def test_sequential_pipeline_with_fault_containment(mock_llm, interceptor, fault_injector):
    pipeline = SequentialPipeline(llm_provider=mock_llm, interceptor=interceptor, fault_injector=fault_injector)
    fi_cfg = {"fault_type": "factual_mutation", "location": "researcher", "seed": 42}

    # Interceptor ON: Should contain error
    res_on = pipeline.run("What is the capital of Australia?", interceptor_enabled=True, fault_injection_config=fi_cfg)
    assert res_on["fault_injected"] is True
    assert len(res_on["handoffs"]) > 0

    # Interceptor OFF: Should propagate unchecked
    res_off = pipeline.run("What is the capital of Australia?", interceptor_enabled=False, fault_injection_config=fi_cfg)
    assert res_off["fault_injected"] is True
    assert len(res_off["handoffs"]) == 0


def test_hierarchical_pipeline(mock_llm, interceptor, fault_injector):
    pipeline = HierarchicalPipeline(llm_provider=mock_llm, interceptor=interceptor, fault_injector=fault_injector)
    res = pipeline.run("Evaluate planetary sizes in the solar system.", interceptor_enabled=True)

    assert "supervisor_plan" in res["agent_outputs"]
    assert "researcher" in res["agent_outputs"]
    assert "specialist" in res["agent_outputs"]
    assert "synthesizer" in res["agent_outputs"]
    assert res["final_output"] is not None


def test_parallel_consensus_pipeline(mock_llm, interceptor, fault_injector):
    pipeline = ParallelConsensusPipeline(llm_provider=mock_llm, interceptor=interceptor, fault_injector=fault_injector)
    fi_cfg = {"fault_type": "factual_mutation", "location": "researcher_a", "seed": 42}
    res = pipeline.run("Compare Transformer self-attention with Mamba state space models.", interceptor_enabled=True, fault_injection_config=fi_cfg)

    assert "researcher_a" in res["agent_outputs"]
    assert "researcher_b" in res["agent_outputs"]
    assert "researcher_c" in res["agent_outputs"]
    assert "consensus_layer" in res["agent_outputs"]
    assert res["final_output"] is not None
