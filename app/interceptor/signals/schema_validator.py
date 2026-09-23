"""Signal 1: Schema Validator using Pydantic."""

from typing import Any, Dict, List, Optional
from pydantic import ValidationError
from app.interceptor.handoff import SignalDetails
from app.llm.base import ObservableReasoningArtifact


class SchemaValidator:
    """Validates structural integrity, field existence, type consistency, and confidence bounds."""

    def __init__(self, strict: bool = True):
        self.strict = strict

    def validate(self, output: Any) -> SignalDetails:
        errors: List[str] = []
        error_count = 0

        if not isinstance(output, dict):
            return SignalDetails(
                signal_name="schema_validator",
                raw_score=1.0,
                normalized_score=1.0,
                weight=0.25,
                passed=False,
                details={"error": "Output is not a valid JSON dictionary"},
                explanations=["Source agent output is not a structured dictionary."],
            )

        # 1. Pydantic validation
        try:
            artifact = ObservableReasoningArtifact.model_validate(output)
        except ValidationError as ve:
            for err in ve.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "Validation error")
                errors.append(f"Schema violation at '{loc}': {msg}")
                error_count += 1
            artifact = None

        # 2. Detailed checks on claims and evidence
        claims = output.get("claims")
        if claims is None:
            errors.append("Missing required 'claims' key.")
            error_count += 2
        elif not isinstance(claims, list):
            errors.append("'claims' must be a list.")
            error_count += 2
        elif len(claims) == 0:
            errors.append("Empty claims list provided by agent.")
            error_count += 1
        else:
            for idx, c in enumerate(claims):
                if not isinstance(c, dict):
                    errors.append(f"Claim #{idx} is not an object.")
                    error_count += 1
                    continue
                
                # Check confidence bounds
                conf = c.get("confidence")
                if conf is None:
                    errors.append(f"Claim #{idx} missing confidence value.")
                    error_count += 1
                elif not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
                    errors.append(f"Claim #{idx} confidence '{conf}' outside valid [0.0, 1.0] interval.")
                    error_count += 1

                # Check evidence field
                ev = c.get("evidence")
                if ev is None:
                    errors.append(f"Claim #{idx} missing 'evidence' list.")
                    error_count += 1
                elif not isinstance(ev, list) or len(ev) == 0:
                    errors.append(f"Claim #{idx} contains no observable evidence.")
                    error_count += 1

        # 3. Answer field check
        answer = output.get("answer")
        if not answer or not isinstance(answer, str) or len(answer.strip()) == 0:
            errors.append("Missing or empty 'answer' field.")
            error_count += 2

        # Compute normalized structural risk score (0.0 to 1.0)
        # 0 errors -> 0.0 risk, >= 5 errors -> 1.0 risk
        normalized_score = min(1.0, error_count * 0.20)
        passed = (error_count == 0)

        return SignalDetails(
            signal_name="schema_validator",
            raw_score=float(error_count),
            normalized_score=normalized_score,
            weight=0.25,
            passed=passed,
            details={
                "schema_valid": passed,
                "schema_error_count": error_count,
                "errors": errors,
            },
            explanations=errors if errors else ["Structured schema integrity fully verified."],
        )
