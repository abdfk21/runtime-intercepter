# Faculty Demonstration Guide

This guide provides a step-by-step walkthrough for presenting the project to an academic evaluation panel.

---

## Demonstration Flow (10–15 Minutes)

### Step 1: Theoretical Motivation & System Architecture (2 mins)
- Explain the **Hallucination Cascade Problem**: In multi-agent LLM systems, minor hallucinations in upstream agents (e.g. Researcher) are accepted as factual premises by downstream agents (Verifier, Synthesizer), causing exponentially compounding cascade errors.
- Open `docs/architecture.md` and show the Mermaid diagram.
- Emphasize the core thesis: Intercepting observable structured handoffs with adaptive multi-signal fusion enables error containment without private chain-of-thought.

### Step 2: Live Terminal Demonstration: Interceptor OFF vs. ON (4 mins)
Run:
```bash
python -m app.cli demo
```
- **Show Condition 1 (OFF)**:
  - Point out that Agent A produces an injected factual error: *"The capital of Australia is Sydney."*
  - Demonstrate how Verifier, Critic, and Synthesizer propagate this error without containment (`Cascade Rate = 100%`).
- **Show Condition 2 (ON)**:
  - Point out that the Interceptor detects the anomaly at Handoff 1.
  - Explain the 3 signals displayed: Schema Score (0.00), Reasoning Score (0.82), NLI Contradiction (0.85).
  - Show the decision: `REPAIR` / `BLOCK`, leading to contained output (`Error Contained = True`).

### Step 3: Adaptive Calibration Demonstration (3 mins)
Run:
```bash
python -m app.cli calibrate --model-name deterministic-mock-v1
```
- Explain to the panel why global fixed thresholds fail across distinct LLMs.
- Show the threshold optimization curve (`results/figures/fig5_calibration_curve.png`).
- Highlight how F1-maximization with precision constraints selects the optimal operating threshold.

### Step 4: Interactive Web Dashboard (4 mins)
Run:
```bash
python -m app.cli dashboard
```
- Open browser at `http://localhost:8000`.
- **Overview Tab**: Show overall containment rate (88.5%), cascade reduction, and latency.
- **Live Trace Viewer**: Type a prompt, check "Inject Fault", click "Run Task", and watch the live handoff cards render with signal scores and decisions in real-time.
- **Model & Architecture Comparison**: Show differences between Sequential, Hierarchical, and Parallel-Consensus topologies.
- **Human Annotation Tab**: Demonstrate how double-annotations are collected to calculate Cohen's Kappa.

### Step 5: Research Statistical Results & Hypothesis Verification (2 mins)
- Open `results/research_summary.md` and `results/tables/table4_interceptor_on_off.md`.
- Present McNemar's test results demonstrating statistical significance ($p < 0.001$) rejecting the null hypothesis $H_0$.
