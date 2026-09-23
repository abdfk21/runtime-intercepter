"""Repair Engine for structured error containment and agent feedback."""

from typing import Any, Callable, Dict, List, Optional, Tuple
from app.core.logging import logger
from app.interceptor.handoff import HandoffInspectionResult, InterceptorDecision


class RepairEngine:
    """Coordinates structured correction cycles for flagged agent handoffs."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def construct_repair_prompt(self, inspection: HandoffInspectionResult) -> str:
        """Construct unambiguous, observable repair feedback without private scratchpads."""
        feedback_lines = [
            "INTERCEPTOR SAFETY NOTICE: The previous output has been flagged during runtime inspection.",
            f"Risk Score: {inspection.risk_score:.2f} (Threshold: {inspection.threshold:.2f})",
            "Identified Anomaly Explanations:",
        ]
        for exp in inspection.explanations:
            feedback_lines.append(f" - {exp}")

        feedback_lines.extend([
            "\nREQUIRED ACTION:",
            "1. Eliminate or ground any unsupported claims.",
            "2. Resolve contradictions against provided context or verified evidence.",
            "3. Adjust confidence metrics to accurately reflect observable evidentiary backing.",
            "4. Return strict structured output containing 'answer', 'claims', 'assumptions', and 'sources'."
        ])
        return "\n".join(feedback_lines)

    def execute_repair_loop(
        self,
        source_agent: Any,
        task_prompt: str,
        initial_output: Dict[str, Any],
        context: Dict[str, Any],
        inspect_fn: Callable[[Dict[str, Any], int], HandoffInspectionResult],
    ) -> Tuple[Dict[str, Any], HandoffInspectionResult, bool]:
        """Runs iterative repair until pass/warning, block, or max attempts reached."""
        current_output = initial_output
        current_inspection = inspect_fn(current_output, 0)
        attempts = 0
        repair_success = False

        while (
            current_inspection.decision == InterceptorDecision.REPAIR
            and attempts < self.max_retries
        ):
            attempts += 1
            logger.info(f"Executing repair attempt {attempts}/{self.max_retries} for agent '{source_agent.name}'")
            repair_prompt = self.construct_repair_prompt(current_inspection)

            # Request agent to re-evaluate with structured feedback
            agent_res = source_agent.execute(
                task_prompt=task_prompt,
                context=context,
                repair_feedback=repair_prompt,
            )
            repaired_output = agent_res.get("output", current_output)

            # Re-inspect repaired output
            new_inspection = inspect_fn(repaired_output, attempts)
            new_inspection.repair_attempt = attempts
            new_inspection.repaired_output = repaired_output

            if new_inspection.decision in [InterceptorDecision.PASS, InterceptorDecision.PASS_WITH_WARNING]:
                repair_success = True
                logger.info(f"Repair attempt {attempts} SUCCEEDED. Risk reduced to {new_inspection.risk_score:.2f}")
                return repaired_output, new_inspection, repair_success
            
            current_output = repaired_output
            current_inspection = new_inspection

        # If exhausted without passing, elevate decision to BLOCK or fallback
        if current_inspection.decision == InterceptorDecision.REPAIR:
            current_inspection.decision = InterceptorDecision.BLOCK
            logger.warning(f"Repair exhausted after {attempts} attempts. Elevating to BLOCK.")

        return current_output, current_inspection, repair_success
