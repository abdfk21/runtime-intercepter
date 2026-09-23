"""Central Runtime Interceptor for multi-agent LLM systems."""

import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.interceptor.handoff import (
    HandoffInspectionResult,
    InterceptorDecision,
    SignalDetails,
)
from app.interceptor.signals.schema_validator import SchemaValidator
from app.interceptor.signals.reasoning_monitor import ReasoningMonitor
from app.interceptor.signals.nli_consistency_checker import NLIConsistencyChecker
from app.interceptor.fusion import SignalFusion
from app.interceptor.decision_engine import DecisionEngine
from app.interceptor.calibration.calibrator import ModelCalibrator
from app.interceptor.calibration.model_profile import ModelCalibrationProfile
from app.interceptor.repair.repair_engine import RepairEngine


class RuntimeInterceptor:
    """Central middleware interceptor operating at multi-agent handoff boundaries."""

    def __init__(
        self,
        mode: Optional[str] = None,
        use_adaptive_calibration: bool = True,
        fixed_threshold: Optional[float] = None,
        enable_repair: bool = True,
    ):
        self.mode = mode or settings.detection_cfg.interceptor_mode
        self.use_adaptive_calibration = use_adaptive_calibration
        self.fixed_threshold = (
            fixed_threshold
            if fixed_threshold is not None
            else settings.detection_cfg.default_threshold
        )
        self.enable_repair = enable_repair

        # Initialize detection signals
        self.schema_validator = SchemaValidator()
        self.reasoning_monitor = ReasoningMonitor()
        self.nli_checker = NLIConsistencyChecker()
        self.fusion = SignalFusion()
        self.decision_engine = DecisionEngine(mode=self.mode)
        self.calibrator = ModelCalibrator()
        self.repair_engine = RepairEngine()

        # Cache of loaded calibration profiles
        self._profiles_cache: Dict[str, Optional[ModelCalibrationProfile]] = {}

    def get_profile(self, model_name: str) -> Optional[ModelCalibrationProfile]:
        if model_name not in self._profiles_cache:
            self._profiles_cache[model_name] = self.calibrator.load_profile(model_name)
        return self._profiles_cache[model_name]

    def inspect_handoff(
        self,
        trace_id: str,
        source_agent: str,
        target_agent: str,
        source_output: Dict[str, Any],
        target_context: Optional[Dict[str, Any]] = None,
        model_name: str = "mock",
        architecture: str = "sequential",
        handoff_id: Optional[str] = None,
        repair_attempt: int = 0,
    ) -> HandoffInspectionResult:
        """Inspects information flowing across an agent handoff boundary."""
        start_time = time.perf_counter()
        handoff_id = handoff_id or str(uuid.uuid4())
        target_context = target_context or {}

        # 1. Execute Signal 1: Schema Validation
        s1_start = time.perf_counter()
        schema_sig = self.schema_validator.validate(source_output)
        s1_ms = (time.perf_counter() - s1_start) * 1000.0

        # 2. Execute Signal 2: Reasoning and Claim Monitoring
        s2_start = time.perf_counter()
        reasoning_sig = self.reasoning_monitor.evaluate(source_output, target_context)
        s2_ms = (time.perf_counter() - s2_start) * 1000.0

        # 3. Execute Signal 3: NLI Cross-Agent Consistency
        s3_start = time.perf_counter()
        nli_sig = self.nli_checker.evaluate(source_output, target_context)
        s3_ms = (time.perf_counter() - s3_start) * 1000.0

        # Determine threshold and weights (Adaptive vs Fixed)
        profile = self.get_profile(model_name) if self.use_adaptive_calibration else None
        if profile and self.use_adaptive_calibration:
            active_threshold = profile.threshold
            warn_threshold = profile.warning_threshold
            block_threshold = profile.block_threshold
            w = profile.weights
            self.fusion.update_weights(
                w.get("schema_weight", 0.25),
                w.get("reasoning_weight", 0.40),
                w.get("consistency_weight", 0.35),
            )
        else:
            active_threshold = self.fixed_threshold
            warn_threshold = settings.detection_cfg.warning_threshold
            block_threshold = settings.detection_cfg.block_threshold

        # 4. Fuse signals
        risk_score, weights_used = self.fusion.fuse(schema_sig, reasoning_sig, nli_sig)

        signals_dict = {
            "schema_validator": schema_sig,
            "reasoning_monitor": reasoning_sig,
            "nli_consistency_checker": nli_sig,
        }

        # 5. Formulate decision
        decision = self.decision_engine.decide(
            risk_score=risk_score,
            signals=signals_dict,
            custom_threshold=active_threshold,
            custom_warning=warn_threshold,
            custom_block=block_threshold,
        )

        all_explanations = []
        for s in [schema_sig, reasoning_sig, nli_sig]:
            all_explanations.extend(s.explanations)

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        result = HandoffInspectionResult(
            trace_id=trace_id,
            handoff_id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            decision=decision,
            risk_score=round(risk_score, 4),
            schema_score=round(schema_sig.normalized_score, 4),
            reasoning_score=round(reasoning_sig.normalized_score, 4),
            consistency_score=round(nli_sig.normalized_score, 4),
            threshold=round(active_threshold, 4),
            signals=signals_dict,
            explanations=all_explanations,
            repair_required=(decision == InterceptorDecision.REPAIR),
            latency_ms=round(total_latency_ms, 2),
            repair_attempt=repair_attempt,
        )

        logger.info(
            f"Handoff {source_agent} -> {target_agent} [{architecture}|{model_name}]: "
            f"Decision={decision.value}, Risk={risk_score:.3f}, Threshold={active_threshold:.3f}",
            extra={
                "trace_id": trace_id,
                "handoff_id": handoff_id,
                "source_agent": source_agent,
                "target_agent": target_agent,
                "model": model_name,
                "architecture": architecture,
                "decision": decision.value,
                "risk_score": risk_score,
                "latency_ms": total_latency_ms,
            },
        )
        return result
