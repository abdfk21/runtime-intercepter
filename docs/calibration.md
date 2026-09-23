# Adaptive Per-Model Calibration

## 1. Rationale
Different foundation models exhibit distinct baseline propensity for hallucination, varying variance in self-reported confidence, and divergent structured reasoning outputs. A single universal fixed threshold either over-flags conservative models (elevating False Positives) or allows hallucination cascades in lenient models (elevating False Negatives).

## 2. Calibration Pipeline
1. **Trace Ingestion**: Ingests held-out calibration traces containing clean baseline generations and controlled fault injections.
2. **Signal Extraction**: Computes normalized $S_{schema}, S_{reasoning}, S_{nli}$ across all handoffs.
3. **Grid Optimization**: Sweeps decision thresholds $\theta \in [0.10, 0.90]$ and weight tuples.
4. **Objective Formulation**: Maximizes $F_1$-score subject to precision constraint $\text{Precision} \ge 0.65$:
   $$\theta^* = \arg\max_{\theta} F_1(\theta) \quad \text{s.t.} \quad \text{Precision}(\theta) \ge 0.65$$
5. **Profile Persistence**: Saves `ModelCalibrationProfile` JSON with empirical thresholds, weights, and error rates.
6. **Data Leakage Prevention**: Calibration is performed strictly on calibration splits; final evaluation occurs on unseen test splits.
