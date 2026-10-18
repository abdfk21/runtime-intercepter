# REST API Reference

The Runtime Interceptor provides a complete REST API accessible by default at `http://localhost:8000`. Interactive OpenAPI documentation is served at `/docs`.

## Endpoints

### 1. `GET /health`
Returns system health, version, active environment, and default LLM provider.

### 2. `POST /run/task`
Executes an end-to-end multi-agent pipeline with optional fault injection and interceptor containment.
- **Request Body**:
  ```json
  {
    "prompt": "What is the capital city of Australia?",
    "architecture": "sequential",
    "model": "mock",
    "interceptor_enabled": true,
    "inject_fault": true,
    "fault_type": "factual_mutation"
  }
  ```

### 3. `POST /intercept`
Direct stateless handoff inspection for custom pipelines.
- **Request Body**:
  ```json
  {
    "trace_id": "tr_12345",
    "source_agent": "researcher",
    "target_agent": "verifier",
    "source_output": { ... },
    "target_context": { ... },
    "model_name": "mock",
    "architecture": "sequential"
  }
  ```

### 4. `POST /calibrate`
Runs adaptive threshold optimization on validation data for a designated model.

### 5. `GET /calibration/{model}`
Retrieves saved `ModelCalibrationProfile` JSON.

### 6. `GET /traces`
Lists recorded multi-agent execution traces with query filters for pagination (`limit`) and model type (`all`, `realtime`, `mock`).

### 7. `GET /traces/{trace_id}`
Retrieves deep trace telemetry for a specific execution, including individual handoffs, signal risk scores, detection explanations, and containment decisions.

### 8. `GET /metrics/summary`
Calculates real-time aggregated metrics across all traces, comparing live vs. mock containment rates, cascade rates, average latency, per-model breakdowns, and provider connectivity status.

### 9. `POST /annotations` & `GET /annotations`
Records and queries human validation ratings.
