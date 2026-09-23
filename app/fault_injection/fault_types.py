"""Enumeration of fault types for controlled hallucination injection."""

from enum import Enum


class FaultType(str, Enum):
    FACTUAL_MUTATION = "factual_mutation"
    NUMERICAL_MUTATION = "numerical_mutation"
    ENTITY_SUBSTITUTION = "entity_substitution"
    DATE_MUTATION = "date_mutation"
    CONTRADICTION_INSERTION = "contradiction_insertion"
    UNSUPPORTED_CLAIM_INSERTION = "unsupported_claim_insertion"
    MISSING_EVIDENCE = "missing_evidence"
    CONFIDENCE_INFLATION = "confidence_inflation"
    CONTEXT_CORRUPTION = "context_corruption"
    SOURCE_MISMATCH = "source_mismatch"
