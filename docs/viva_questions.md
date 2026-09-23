# Comprehensive Viva Voce Questions & Answers

### 1. What is a hallucination cascade in multi-agent LLM systems?
**Answer:** In single-agent setups, a hallucination is isolated to the immediate output. In multi-agent pipelines (e.g. sequential or hierarchical graphs), an upstream agent's hallucination is passed as trusted context or premise to subsequent agents. Downstream agents elaborate, rationalize, and build on that false premise, compounding the error until the final synthesis is completely ungrounded. This propagation dynamic is termed a *hallucination cascade*.

### 2. Why intercept at agent handoff boundaries rather than checking only the final output?
**Answer:** Detecting errors at the final stage suffers from the "attribution problem": by the time an error reaches the final synthesis, multiple agents have corroborated and blended it, making diagnosis difficult and requiring a full re-run of the entire multi-agent pipeline. Intercepting at handoff boundaries allows early containment, localized repair feedback directly to the responsible agent, and prevents downstream compute waste.

### 3. Why did you choose not to inspect hidden chain-of-thought (CoT)?
**Answer:** Proprietary models (like GPT-4o or Gemini Flash) frequently mask or do not expose hidden reasoning tokens. Relying on hidden CoT creates an unobservable dependency that lacks standardized formats and risks exposing private prompt techniques. Our approach requires agents to produce observable structured reasoning artifacts (explicit claims, self-reported confidence, and cited evidence), ensuring verifiable inspection across any LLM backend.

### 4. What are the three detection signals and why are they orthogonal?
**Answer:**
1. **Schema Validator (Structural)**: Ensures JSON validity, required fields, and bound checks using Pydantic. It catches structural breakdowns and format degradation.
2. **Reasoning & Claim Monitor (Semantic & Evidentiary)**: Checks whether atomic claims are semantically grounded in observable evidence and flags confidence inflation or ungrounded numbers.
3. **NLI Consistency Checker (Logical/Relational)**: Uses Natural Language Inference to evaluate entailment, neutrality, or contradiction between upstream premise and downstream claims.
They are orthogonal because an output can be grammatically and structurally valid while logically contradicting prior evidence, or logically consistent but lacking structural grounding.

### 5. Why is adaptive per-model calibration necessary instead of a single fixed threshold?
**Answer:** LLMs exhibit different baseline calibration characteristics. For example, some models tend to report overly confident confidence scores (0.95+) on vague claims, while others are calibrated conservatively. A fixed global threshold of 0.50 might produce an unacceptable false positive rate on one model while failing to catch severe cascades in another. Adaptive calibration tunes the operating decision threshold $\theta^*$ on held-out validation data for each model to optimize a target objective (e.g. $F_1$ score).

### 6. How do you prevent data leakage during calibration?
**Answer:** We enforce a strict stratified partition between the calibration set and the test set using random seeding and label stratification. Calibration traces are used strictly to optimize the threshold and signal weights. The final benchmark evaluation is executed on completely held-out, unseen test traces.

### 7. What statistical test did you choose to compare Interceptor ON vs. OFF, and why?
**Answer:** We use **McNemar's test** for paired categorical data. Because the exact same tasks and seeds are evaluated under both conditions (INTERCEPTOR_ON vs. INTERCEPTOR_OFF), the binary outcome (error contained: Yes/No) is inherently paired. Standard two-sample chi-square or independent t-tests would violate the independence assumption. McNemar's test specifically examines the discordant pairs ($b$ and $c$ in the 2x2 contingency table) to determine whether interception significantly increases containment.

### 8. What is Cohen's Kappa and why is it used in human validation?
**Answer:** Cohen's Kappa ($\kappa$) measures inter-annotator agreement between two independent raters on categorical classifications, correcting for agreement that could happen by chance:
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$
where $P_o$ is observed agreement and $P_e$ is expected chance agreement. It validates that our benchmark dataset labels (VALID, SUSPICIOUS, HALLUCINATED, CONTRADICTORY) represent reproducible human consensus rather than arbitrary subjective ratings.

### 9. How does error cascade behavior differ across the three architectures?
**Answer:**
- **Sequential**: Highest vulnerability to deep cascade propagation (error depth up to 3), as each agent directly consumes the previous agent's unvalidated output.
- **Hierarchical**: The supervisor acts as a natural hub, but if the supervisor itself accepts a corrupted specialist report, the error is immediately stamped into the authoritative consolidated brief.
- **Parallel-Consensus**: Most resilient to individual agent faults; unless multiple parallel researchers experience the exact same fault, majority voting and consensus filtering isolate the anomaly before synthesis.

### 10. Does your system claim to determine whether a claim is absolute truth?
**Answer:** Absolutely not. We explicitly follow epistemic humility principles: the interceptor calculates risk scores, anomaly metrics, and premise-evidence alignment relative to observable context. It detects suspected hallucinations, evidentiary mismatches, and logical contradictions. It never claims infallible ontological truth.
