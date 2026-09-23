"""Deterministic controlled fault injector for error containment benchmarking."""

import copy
import random
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from app.fault_injection.fault_types import FaultType


class FaultInjectionResult:
    """Encapsulates reproducible fault injection metadata."""

    def __init__(
        self,
        fault_id: str,
        fault_type: FaultType,
        original_output: Dict[str, Any],
        corrupted_output: Dict[str, Any],
        location: str,
        severity: float,
        ground_truth: str,
        random_seed: int,
    ):
        self.fault_id = fault_id
        self.fault_type = fault_type
        self.original_output = original_output
        self.corrupted_output = corrupted_output
        self.location = location
        self.severity = severity
        self.ground_truth = ground_truth
        self.random_seed = random_seed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "original_output": self.original_output,
            "corrupted_output": self.corrupted_output,
            "location": self.location,
            "severity": self.severity,
            "ground_truth": self.ground_truth,
            "random_seed": self.random_seed,
        }


class FaultInjector:
    """Injects controlled, deterministic factual anomalies and structural corruptions."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def inject_fault(
        self,
        output: Dict[str, Any],
        fault_type: Optional[FaultType] = None,
        location: str = "agent_handoff",
        severity: float = 0.8,
        seed: Optional[int] = None,
    ) -> FaultInjectionResult:
        active_seed = seed if seed is not None else self.seed
        rng = random.Random(active_seed)
        fault_id = f"flt_{uuid.uuid4().hex[:10]}"

        if fault_type is None:
            fault_type = rng.choice(list(FaultType))

        original = copy.deepcopy(output)
        corrupted = copy.deepcopy(output)
        claims = corrupted.get("claims", [])
        ground_truth = "Verified factual baseline state"

        # Apply specific fault logic
        if fault_type == FaultType.FACTUAL_MUTATION:
            if claims:
                idx = rng.randint(0, len(claims) - 1)
                orig_claim = claims[idx].get("claim", "")
                ground_truth = orig_claim
                # Swap factual assertion
                mutated = orig_claim.replace("Canberra", "Sydney").replace("Jupiter", "Mars").replace("capital", "commercial port")
                if mutated == orig_claim:
                    mutated = f"Contrary to consensus, {orig_claim} is demonstrably refuted by recent anomalies."
                claims[idx]["claim"] = mutated
                corrupted["answer"] = f"Mutated finding: {mutated}"
            else:
                corrupted["answer"] = "Factual mutation: The capital of Australia is Sydney."
                claims.append({"claim": "Sydney is the federal capital of Australia.", "confidence": 0.99, "evidence": ["Census 2021"]})

        elif fault_type == FaultType.NUMERICAL_MUTATION:
            # Find numbers and corrupt them drastically
            if claims:
                idx = rng.randint(0, len(claims) - 1)
                orig_claim = claims[idx].get("claim", "")
                ground_truth = orig_claim
                nums = re.findall(r"\b\d+\b", orig_claim)
                if nums:
                    target_num = nums[0]
                    corrupt_num = str(int(target_num) * 10 + 7)
                    claims[idx]["claim"] = orig_claim.replace(target_num, corrupt_num, 1)
                else:
                    claims[idx]["claim"] = f"{orig_claim} with 9942 unexplained units."
            corrupted["answer"] = corrupted.get("answer", "") + " Mutated quantity: 994200 units observed."

        elif fault_type == FaultType.ENTITY_SUBSTITUTION:
            entities = [
                ("Australia", "Austria"), ("Jupiter", "Saturn"), ("DNA", "RNA"),
                ("Einstein", "Newton"), ("Python", "Perl"), ("Google", "Yahoo")
            ]
            sub = rng.choice(entities)
            if claims:
                idx = rng.randint(0, len(claims) - 1)
                orig = claims[idx].get("claim", "")
                ground_truth = orig
                claims[idx]["claim"] = orig.replace(sub[0], sub[1]) if sub[0] in orig else f"{sub[1]} substituted for {orig}"
            corrupted["answer"] = corrupted.get("answer", "").replace(sub[0], sub[1])

        elif fault_type == FaultType.DATE_MUTATION:
            orig_ans = corrupted.get("answer", "")
            ground_truth = orig_ans
            corrupted["answer"] = re.sub(r"\b(19\d\d|20\d\d)\b", "1492", orig_ans)
            if claims:
                for c in claims:
                    c["claim"] = re.sub(r"\b(19\d\d|20\d\d)\b", "1492", c.get("claim", ""))

        elif fault_type == FaultType.CONTRADICTION_INSERTION:
            # Insert a claim directly opposing an existing claim
            if claims:
                orig = claims[0].get("claim", "")
                ground_truth = orig
                contra = f"However, it is strictly confirmed that {orig} is entirely false and contradicted by observation."
                claims.append({"claim": contra, "confidence": 0.95, "evidence": ["Counter-observation registry"]})
            else:
                claims.append({"claim": "Statement is true.", "confidence": 0.9, "evidence": ["Ref"]})
                claims.append({"claim": "Statement is completely false.", "confidence": 0.9, "evidence": ["Ref"]})

        elif fault_type == FaultType.UNSUPPORTED_CLAIM_INSERTION:
            unsupported = "All quantum gravity experiments conclusively confirmed tachyon teleportation at room temperature."
            ground_truth = "Tachyon teleportation is unproven theoretical conjecture"
            claims.append({
                "claim": unsupported,
                "confidence": 0.99,
                "evidence": []  # Empty evidence
            })
            corrupted["answer"] = corrupted.get("answer", "") + f" {unsupported}"

        elif fault_type == FaultType.MISSING_EVIDENCE:
            # Strip all evidence from claims
            ground_truth = "Claims possessed grounding evidence"
            for c in claims:
                c["evidence"] = []

        elif fault_type == FaultType.CONFIDENCE_INFLATION:
            # Inflate confidence to 1.0 on dubious or empty evidence
            ground_truth = "Calibrated confidence score"
            for c in claims:
                c["confidence"] = 1.0
                if len(c.get("evidence", [])) > 1:
                    c["evidence"] = [c["evidence"][0]]

        elif fault_type == FaultType.CONTEXT_CORRUPTION:
            ground_truth = "Standard context environment"
            corrupted["assumptions"] = ["Assumed that standard laws of thermodynamics are inverted in this scope."]
            if claims:
                claims[0]["claim"] = f"Under inverted thermodynamics: {claims[0].get('claim', '')}"

        elif fault_type == FaultType.SOURCE_MISMATCH:
            ground_truth = "Peer-reviewed academic sources"
            corrupted["sources"] = ["Unverified anonymous forum blog post #40291"]
            if claims:
                claims[0]["evidence"] = ["Anonymous internet post claim."]

        corrupted["claims"] = claims
        return FaultInjectionResult(
            fault_id=fault_id,
            fault_type=fault_type,
            original_output=original,
            corrupted_output=corrupted,
            location=location,
            severity=severity,
            ground_truth=ground_truth,
            random_seed=active_seed,
        )
