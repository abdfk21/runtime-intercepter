"""Signal Fusion combining schema, reasoning, and consistency signals into unified risk score."""

from typing import Dict, Tuple
from app.interceptor.handoff import SignalDetails


class SignalFusion:
    """Combines heterogeneous detection signals into a calibrated composite risk score."""

    def __init__(
        self,
        schema_weight: float = 0.25,
        reasoning_weight: float = 0.40,
        consistency_weight: float = 0.35,
    ):
        total = schema_weight + reasoning_weight + consistency_weight
        self.w_schema = schema_weight / total
        self.w_reasoning = reasoning_weight / total
        self.w_consistency = consistency_weight / total

    def update_weights(self, schema_weight: float, reasoning_weight: float, consistency_weight: float):
        total = schema_weight + reasoning_weight + consistency_weight
        if total > 0:
            self.w_schema = schema_weight / total
            self.w_reasoning = reasoning_weight / total
            self.w_consistency = consistency_weight / total

    def fuse(
        self,
        schema_signal: SignalDetails,
        reasoning_signal: SignalDetails,
        consistency_signal: SignalDetails,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute composite risk score: in range [0.0, 1.0]."""
        s_score = schema_signal.normalized_score
        r_score = reasoning_signal.normalized_score
        c_score = consistency_signal.normalized_score

        composite_risk = (
            self.w_schema * s_score
            + self.w_reasoning * r_score
            + self.w_consistency * c_score
        )
        composite_risk = max(0.0, min(1.0, composite_risk))

        weights_used = {
            "schema_weight": round(self.w_schema, 4),
            "reasoning_weight": round(self.w_reasoning, 4),
            "consistency_weight": round(self.w_consistency, 4),
        }
        return composite_risk, weights_used
