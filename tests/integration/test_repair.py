"""Integration tests for the Repair Engine."""

import pytest
from app.interceptor.repair.repair_engine import RepairEngine
from app.interceptor.handoff import HandoffInspectionResult, InterceptorDecision, SignalDetails
from app.agents.researcher import ResearcherAgent


def test_repair_engine_prompt_construction():
    engine = RepairEngine(max_retries=2)
    inspection = HandoffInspectionResult(
        trace_id="tr_test",
        handoff_id="h_test",
        decision=InterceptorDecision.REPAIR,
        risk_score=0.75,
        schema_score=0.1,
        reasoning_score=0.8,
        consistency_score=0.7,
        threshold=0.5,
        explanations=["Claim lacks observable evidentiary grounding."],
    )

    prompt = engine.construct_repair_prompt(inspection)
    assert "INTERCEPTOR SAFETY NOTICE" in prompt
    assert "Claim lacks observable evidentiary grounding." in prompt
    assert "REQUIRED ACTION:" in prompt


def test_repair_engine_retry_loop(mock_llm):
    agent = ResearcherAgent(llm_provider=mock_llm)
    engine = RepairEngine(max_retries=2)

    initial_output = {
        "answer": "Unrepaired draft",
        "claims": [{"claim": "Suspect claim", "confidence": 0.99, "evidence": []}],
        "assumptions": [],
        "sources": []
    }

    # Inspect function that approves repair on attempt 1
    def fake_inspect(out, attempt):
        dec = InterceptorDecision.PASS if attempt >= 1 else InterceptorDecision.REPAIR
        return HandoffInspectionResult(
            trace_id="tr_1",
            handoff_id="h_1",
            decision=dec,
            risk_score=0.2 if dec == InterceptorDecision.PASS else 0.8,
            schema_score=0.0,
            reasoning_score=0.2,
            consistency_score=0.2,
            threshold=0.5,
            explanations=[],
            repair_attempt=attempt,
        )

    repaired, final_insp, success = engine.execute_repair_loop(
        source_agent=agent,
        task_prompt="Verify claim",
        initial_output=initial_output,
        context={},
        inspect_fn=fake_inspect,
    )

    assert success is True
    assert final_insp.decision == InterceptorDecision.PASS
    assert final_insp.repair_attempt == 1
