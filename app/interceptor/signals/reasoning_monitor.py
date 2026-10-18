"""Signal 2: Reasoning and Claim Monitor."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.core.ml_compat import semantic_similarity_fallback
from app.interceptor.handoff import SignalDetails

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - fallback for locked-down Windows environments
    TfidfVectorizer = None
    cosine_similarity = None


class ReasoningMonitor:
    """Monitors observable claims for evidence support, confidence inflation, numerical alignment, and semantic cohesion."""

    def __init__(
        self,
        min_evidence_similarity: float = 0.18,
        max_unsupported_ratio: float = 0.25,
        detect_excessive_certainty: bool = True,
        detect_numerical_hallucinations: bool = True,
    ):
        self.min_evidence_similarity = min_evidence_similarity
        self.max_unsupported_ratio = max_unsupported_ratio
        self.detect_excessive_certainty = detect_excessive_certainty
        self.detect_numerical_hallucinations = detect_numerical_hallucinations

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract integers, floats, and percentages from text."""
        return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))

    def _compute_semantic_similarity(self, query: str, references: List[str]) -> float:
        """Compute highest cosine similarity between query and candidate references using char/word n-gram TF-IDF."""
        if not query or not references:
            return 0.0
        
        valid_refs = [r for r in references if isinstance(r, str) and len(r.strip()) > 0]
        if not valid_refs:
            return 0.0

        corpus = [query] + valid_refs
        try:
            if TfidfVectorizer is None or cosine_similarity is None:
                raise RuntimeError("scikit-learn is unavailable")
            vectorizer = TfidfVectorizer(ngram_range=(2, 4), analyzer="char_wb")
            tfidf = vectorizer.fit_transform(corpus)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
            val = float(np.max(sims)) if len(sims) > 0 else 0.0
            return max(0.0, min(1.0, val))
        except Exception:
            val = float(semantic_similarity_fallback(query, valid_refs))
            return max(0.0, min(1.0, val))

    def evaluate(self, output: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> SignalDetails:
        context = context or {}
        claims: List[Dict[str, Any]] = output.get("claims", [])
        answer: str = output.get("answer", "")
        assumptions: List[str] = output.get("assumptions", [])
        
        # Collect external context text recursively
        context_texts: List[str] = []
        def _extract_ctx(v: Any):
            if isinstance(v, str):
                if v.strip():
                    context_texts.append(v.strip())
            elif isinstance(v, dict):
                if "answer" in v and isinstance(v["answer"], str):
                    context_texts.append(v["answer"].strip())
                if "claims" in v and isinstance(v["claims"], list):
                    for c in v["claims"]:
                        if isinstance(c, dict) and "claim" in c:
                            context_texts.append(str(c["claim"]).strip())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    _extract_ctx(item)

        if isinstance(context, dict):
            for k, v in context.items():
                _extract_ctx(v)

        issues: List[str] = []
        unsupported_count = 0
        inflated_confidence_count = 0
        numerical_mismatch_count = 0
        internal_contradiction_count = 0

        # All evidence pooled
        pooled_evidence: List[str] = []
        for c in claims:
            if isinstance(c, dict):
                ev_list = c.get("evidence", [])
                if isinstance(ev_list, list):
                    pooled_evidence.extend([str(e) for e in ev_list])

        all_references = pooled_evidence + context_texts

        if not claims:
            issues.append("No observable claims present to monitor reasoning validity.")
            return SignalDetails(
                signal_name="reasoning_monitor",
                raw_score=0.8,
                normalized_score=0.8,
                weight=0.40,
                passed=False,
                details={"unsupported_claims": 0, "total_claims": 0},
                explanations=issues,
            )

        # Evaluate each claim
        claim_texts: List[str] = []
        for i, c_obj in enumerate(claims):
            if not isinstance(c_obj, dict):
                continue
            claim_text = c_obj.get("claim", "")
            confidence = float(c_obj.get("confidence", 0.5))
            evidence_items = [str(e) for e in c_obj.get("evidence", []) if isinstance(e, str)]
            claim_texts.append(claim_text)

            # 1. Semantic support check (against evidence, context, and external sources - not the ungrounded answer)
            grounding_refs = evidence_items + context_texts + [s for s in output.get("sources", []) if isinstance(s, str)]
            sim_score = self._compute_semantic_similarity(claim_text, grounding_refs)
            if sim_score < self.min_evidence_similarity or (len(evidence_items) == 0 and not context_texts):
                unsupported_count += 1
                issues.append(f"Claim #{i+1} has low observable evidence alignment (similarity: {sim_score:.2f}).")

            # 2. Confidence / Evidence mismatch (Confidence Inflation)
            if self.detect_excessive_certainty:
                if confidence > 0.90 and (len(evidence_items) == 0 or sim_score < 0.20):
                    inflated_confidence_count += 1
                    issues.append(f"Claim #{i+1} displays excessive certainty ({confidence:.2f}) despite weak grounding.")

            # 3. Numerical hallucination check
            if self.detect_numerical_hallucinations:
                claim_nums = self._extract_numbers(claim_text)
                if claim_nums and all_references:
                    ref_text_combined = " ".join(all_references)
                    ref_nums = self._extract_numbers(ref_text_combined)
                    unsupported_nums = claim_nums - ref_nums
                    if unsupported_nums:
                        numerical_mismatch_count += 1
                        issues.append(
                            f"Claim #{i+1} introduces ungrounded numerical figures: {unsupported_nums}"
                        )

        # 4. Internal contradiction check between claims
        for i in range(len(claim_texts)):
            for j in range(i + 1, len(claim_texts)):
                c1, c2 = claim_texts[i].lower(), claim_texts[j].lower()
                # Antonym / direct conflict patterns
                conflicts = [
                    ("is", "is not"), ("can", "cannot"), ("increased", "decreased"),
                    ("always", "never"), ("capital", "not the capital"), ("true", "false"),
                    ("before", "after")
                ]
                for pos, neg in conflicts:
                    if (pos in c1 and neg in c2) or (pos in c2 and neg in c1):
                        internal_contradiction_count += 1
                        issues.append(f"Potential contradiction between Claim #{i+1} and Claim #{j+1}.")
                        break

        total_claims = max(1, len(claims))
        unsupported_ratio = unsupported_count / total_claims
        
        # Risk score calculation: scale sensitively when anomalies are identified
        raw_risk = 0.05
        if unsupported_count > 0:
            raw_risk = max(raw_risk, 0.45 + 0.45 * unsupported_ratio)
        if numerical_mismatch_count > 0:
            raw_risk = max(raw_risk, 0.70)
        if internal_contradiction_count > 0:
            raw_risk = max(raw_risk, 0.80)
        if inflated_confidence_count > 0:
            raw_risk = max(raw_risk, 0.50)

        normalized_score = float(np.clip(raw_risk, 0.0, 1.0))
        passed = (unsupported_count == 0 and numerical_mismatch_count == 0 and internal_contradiction_count == 0)

        return SignalDetails(
            signal_name="reasoning_monitor",
            raw_score=raw_risk,
            normalized_score=normalized_score,
            weight=0.40,
            passed=passed,
            details={
                "total_claims": total_claims,
                "unsupported_claims": unsupported_count,
                "inflated_confidence_count": inflated_confidence_count,
                "numerical_mismatch_count": numerical_mismatch_count,
                "internal_contradictions": internal_contradiction_count,
            },
            explanations=issues if issues else ["Observable claims and evidence are logically grounded."],
        )
