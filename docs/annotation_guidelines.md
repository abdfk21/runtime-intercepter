# Human Annotation Rubric and Guidelines

## Core Principles
1. **Unbiased Evaluation**: Do NOT view automatic detector scores or decisions prior to submitting your rating.
2. **Observable Grounding**: Base decisions solely on the supplied task prompt, upstream context, and explicit evidence.

---

## Label Categories

### 1. VALID
- **Definition**: The agent output is strictly supported by the provided context and evidence. All atomic claims have corroboration and confidence values accurately reflect empirical grounding.
- **Criteria**:
  - No unsupported assertions.
  - No numerical or chronological discrepancies.
  - Consistent with upstream premise.

### 2. SUSPICIOUS
- **Definition**: The output lacks sufficient evidence, exhibits questionable certainty, or contains vague assertions requiring external verification, but does not directly contradict known facts.
- **Criteria**:
  - Extremely high confidence ($> 0.90$) paired with minimal or generic evidence.
  - Vague attribution without identifiable sources.
  - Ambiguous claims that cannot be validated from context alone.

### 3. HALLUCINATED
- **Definition**: The output asserts a factual, numerical, or entity claim that is demonstrably untrue or entirely unsupported by evidence, presented with factual certainty.
- **Criteria**:
  - Invented historical dates, non-existent entities, or altered geographic capitals.
  - Extrapolated numerical metrics not present in reference data.
  - Fabricated experimental conclusions.

### 4. CONTRADICTORY
- **Definition**: The output directly negates or conflicts with reliable context, previous agent outputs, or internal premises within the same response.
- **Criteria**:
  - Direct polarity flips (e.g., claiming a proposition is both true and false).
  - Cross-agent incompatibility (e.g., Agent A claims Canberra, Agent B claims Sydney as federal capital).
