# System Limitations and Epistemic Boundaries

## 1. Risk and Anomaly Detection vs. Absolute Truth
The Runtime Interceptor is strictly an anomaly, risk detection, and error containment middleware.
- **The system CANNOT and DOES NOT claim to determine whether a statement is ontologically true or false.**
- Detector scores represent computed risk metrics ($\text{risk\_score} \in [0.0, 1.0]$) reflecting statistical likelihood of evidentiary disconnect, structural anomaly, or premise contradiction.
- High risk signals indicate a *suspected hallucination*, *unsupported claim*, or *evidence conflict* relative to observable context.

## 2. Reliance on Observable Structured Artifacts
- The reasoning monitor evaluates explicit claims, confidence declarations, and evidence lists.
- If an agent generates free-form unstructured prose without claims, detection sensitivity decreases.
- Hidden internal chain-of-thought is neither inspected nor relied upon.

## 3. Latency Overhead Trade-Offs
- Evaluating NLI premise-hypothesis pairs introduces bounded latency overhead (typically 20ms to 50ms per handoff).
- In high-throughput, latency-critical applications, lightweight heuristic mode or async batched inference may be preferable.

## 4. Domain Generalization
- Calibrated thresholds reflect empirical validation distributions. Severe distribution shifts into unfamiliar technical jargon may require domain re-calibration.
