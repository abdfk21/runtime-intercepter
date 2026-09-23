"""Decision Engine for mapping calibrated risk scores to interceptor actions."""

from typing import Dict, List, Optional
from app.interceptor.handoff import InterceptorDecision, SignalDetails


class DecisionEngine:
    """Evaluates composite risk score and individual signals against calibrated thresholds."""

    def __init__(
        self,
        mode: str = "active",
        default_threshold: float = 0.50,
        warning_threshold: float = 0.35,
        block_threshold: float = 0.80,
    ):
        self.mode = mode.lower()  # "active" or "passive"
        self.default_threshold = default_threshold
        self.warning_threshold = warning_threshold
        self.block_threshold = block_threshold

    def decide(
        self,
        risk_score: float,
        signals: Dict[str, SignalDetails],
        custom_threshold: Optional[float] = None,
        custom_warning: Optional[float] = None,
        custom_block: Optional[float] = None,
    ) -> InterceptorDecision:
        threshold = custom_threshold if custom_threshold is not None else self.default_threshold
        warn_th = custom_warning if custom_warning is not None else self.warning_threshold
        block_th = custom_block if custom_block is not None else self.block_threshold

        # In passive mode, interceptor observes and logs without intervention
        if self.mode == "passive":
            if risk_score >= threshold:
                return InterceptorDecision.PASS_WITH_WARNING
            return InterceptorDecision.PASS

        # Active containment mode
        if risk_score >= block_th:
            return InterceptorDecision.BLOCK

        if risk_score >= threshold:
            # Check if repairable
            schema_sig = signals.get("schema_validator")
            if schema_sig and schema_sig.normalized_score > 0.8:
                # Severe structural failure: attempt repair
                return InterceptorDecision.REPAIR
            return InterceptorDecision.REPAIR

        if risk_score >= warn_th:
            return InterceptorDecision.PASS_WITH_WARNING

        return InterceptorDecision.PASS
