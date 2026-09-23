# Methodology

## 1. Detection Signals

### Signal 1: Schema Validation (Pydantic)
- **Objective**: Evaluates structural completeness, JSON integrity, and adherence to observable reasoning schemas.
- **Failures Detected**: Missing required fields, unparseable types, missing claims, empty evidence lists, confidence values outside $[0.0, 1.0]$.
- **Normalization**: Error counts mapped smoothly to $[0.0, 1.0]$ via $S_{schema} = \min(1.0, k \cdot 0.20)$.
- **Principle**: Schema failure represents structural risk, not an absolute ground-truth hallucination.

### Signal 2: Reasoning and Claim Monitoring
- **Objective**: Inspects atomic claims for evidence support, confidence inflation, and numerical consistency.
- **Components**:
  1. *Semantic Alignment*: Cosine similarity between claim and observable evidence/context using n-gram vectorization.
  2. *Confidence Inflation*: Penalizes claims with $\text{confidence} > 0.90$ lacking empirical support.
  3. *Numerical Hallucination*: Flags numbers in claims absent from prior context or evidence.
  4. *Internal Contradiction*: Detects polarity flips or antonym conflicts across claims within the same response.

### Signal 3: NLI Cross-Agent Consistency
- **Objective**: Evaluates cross-agent handoffs using Natural Language Inference (Premise vs. Hypothesis).
- **Inference Classes**: $\text{Entailment}$, $\text{Neutral}$, $\text{Contradiction}$.
- **Risk Score Mapping**: Contradiction probabilities directly elevate the consistency risk score $S_{nli}$.

---

## 2. Signal Fusion
Signals are combined linearly into a composite risk score:
$$R = w_s \cdot S_{schema} + w_r \cdot S_{reasoning} + w_c \cdot S_{nli}$$
where $\sum w_i = 1.0$. Default weights are $(0.25, 0.40, 0.35)$ prior to calibration.

---

## 3. Decision Boundary Mapping
Decisions are mapped based on calibrated thresholds:
- $\text{PASS}$: $R < \theta_{warn}$
- $\text{PASS\_WITH\_WARNING}$: $\theta_{warn} \le R < \theta$
- $\text{REPAIR}$: $\theta \le R < \theta_{block}$ (Triggers structured correction cycle)
- $\text{BLOCK}$: $R \ge \theta_{block}$ (Suppresses corrupted output, emits safe containment message)
