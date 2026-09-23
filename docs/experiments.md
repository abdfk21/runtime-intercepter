# Experimental Design and Benchmarking

## 1. Experimental Matrix
The benchmarking system evaluates:
- **3 LLM Backends**: Google Gemini (`gemini-2.5-flash`), Groq (`llama-3.3-70b-versatile`), Offline Mock (`deterministic-mock-v1`).
- **3 Multi-Agent Topologies**: Sequential, Hierarchical, Parallel-Consensus.
- **2 Experimental Conditions**: `INTERCEPTOR_ON` vs. `INTERCEPTOR_OFF`.
- **10 Controlled Fault Injections**:
  1. `factual_mutation`
  2. `numerical_mutation`
  3. `entity_substitution`
  4. `date_mutation`
  5. `contradiction_insertion`
  6. `unsupported_claim_insertion`
  7. `missing_evidence`
  8. `confidence_inflation`
  9. `context_corruption`
  10. `source_mismatch`

## 2. Research Hypotheses
- **H1**: Adaptive per-model calibration reduces the cross-model effectiveness gap compared with a fixed global threshold.
- **H2**: Runtime interception reduces hallucination cascade propagation compared with interceptor OFF.
- **H3**: Cascade behavior differs between sequential, hierarchical and parallel-consensus architectures.
- **H4**: Combining schema validation, reasoning monitoring and NLI consistency performs differently from individual signals alone.
