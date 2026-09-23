// Dashboard Interactive Logic

function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  
  event.currentTarget.classList.add('active');
}

// Initialize Charts
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  loadCalibrationProfiles();
});

function initCharts() {
  // Chart 1: ON vs OFF
  const ctxOnOff = document.getElementById('chart-on-off');
  if (ctxOnOff) {
    new Chart(ctxOnOff, {
      type: 'bar',
      data: {
        labels: ['Cascade Occurrence', 'Error Containment'],
        datasets: [
          { label: 'Interceptor OFF', data: [79.7, 20.3], backgroundColor: '#ef4444' },
          { label: 'Interceptor ON', data: [11.5, 88.5], backgroundColor: '#22c55e' }
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
    });
  }

  // Chart 2: Cascade Depth
  const ctxDepth = document.getElementById('chart-depth');
  if (ctxDepth) {
    new Chart(ctxDepth, {
      type: 'doughnut',
      data: {
        labels: ['Contained at Handoff 1', 'Contained at Handoff 2', 'Reached Final Output'],
        datasets: [{
          data: [65, 23, 12],
          backgroundColor: ['#22c55e', '#3b82f6', '#ef4444']
        }]
      },
      options: { responsive: true }
    });
  }

  // Chart 3: Model Comparison
  const ctxModels = document.getElementById('chart-models');
  if (ctxModels) {
    new Chart(ctxModels, {
      type: 'bar',
      data: {
        labels: ['deterministic-mock-v1', 'gemini-2.5-flash', 'llama-3.3-70b-versatile'],
        datasets: [
          { label: 'Fixed Threshold F1', data: [0.74, 0.77, 0.75], backgroundColor: '#64748b' },
          { label: 'Adaptive Calibrated F1', data: [0.89, 0.91, 0.88], backgroundColor: '#3b82f6' }
        ]
      },
      options: { responsive: true, scales: { y: { min: 0.5, max: 1.0 } } }
    });
  }

  // Chart 4: Architecture Comparison
  const ctxArch = document.getElementById('chart-architectures');
  if (ctxArch) {
    new Chart(ctxArch, {
      type: 'bar',
      data: {
        labels: ['Sequential', 'Hierarchical', 'Parallel-Consensus'],
        datasets: [
          { label: 'Unchecked Cascade (OFF)', data: [83.3, 76.5, 68.2], backgroundColor: '#ef4444' },
          { label: 'Error Containment (ON)', data: [87.5, 89.2, 92.4], backgroundColor: '#22c55e' }
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
    });
  }

  // Chart 5: Calibration curve
  const ctxCalib = document.getElementById('chart-calibration-curve');
  if (ctxCalib) {
    new Chart(ctxCalib, {
      type: 'line',
      data: {
        labels: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        datasets: [
          { label: 'F1 Score', data: [0.65, 0.74, 0.86, 0.89, 0.82, 0.71, 0.55], borderColor: '#3b82f6', tension: 0.3 },
          { label: 'Precision', data: [0.55, 0.68, 0.81, 0.92, 0.94, 0.96, 0.98], borderColor: '#22c55e', borderDash: [5, 5] },
          { label: 'Recall', data: [0.95, 0.92, 0.88, 0.82, 0.72, 0.56, 0.40], borderColor: '#ef4444', borderDash: [2, 2] }
        ]
      },
      options: { responsive: true }
    });
  }

  // Chart 6: Ablation
  const ctxAbl = document.getElementById('chart-ablation');
  if (ctxAbl) {
    new Chart(ctxAbl, {
      type: 'bar',
      data: {
        labels: ['Full System', 'Schema Only', 'Reasoning Only', 'NLI Only', 'Schema + Reasoning', 'Schema + NLI', 'Reasoning + NLI'],
        datasets: [{
          label: 'F1 Detection Score',
          data: [0.895, 0.584, 0.742, 0.718, 0.795, 0.768, 0.841],
          backgroundColor: '#3b82f6'
        }]
      },
      options: { indexAxis: 'y', responsive: true, scales: { x: { min: 0.4, max: 1.0 } } }
    });
  }
}

async function runLiveTask() {
  const prompt = document.getElementById('interactive-prompt').value;
  const arch = document.getElementById('interactive-arch').value;
  const fault = document.getElementById('interactive-fault').checked;
  const container = document.getElementById('live-trace-results');

  container.innerHTML = '<div class="flow-placeholder">Executing multi-agent pipeline and runtime interceptor...</div>';

  try {
    const res = await fetch('/run/task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        architecture: arch,
        model: 'mock',
        interceptor_enabled: true,
        inject_fault: fault,
        fault_type: 'factual_mutation'
      })
    });
    const data = await res.json();
    renderTraceFlow(data, container);
  } catch (err) {
    container.innerHTML = `<div class="flow-placeholder text-danger">Error: ${err.message}</div>`;
  }
}

function renderTraceFlow(data, container) {
  let html = '';
  const handoffs = data.handoffs || [];

  if (handoffs.length === 0) {
    html += '<div class="flow-step">Pipeline completed without intermediate handoff flags.</div>';
  }

  handoffs.forEach((h, idx) => {
    const isAlert = h.decision !== 'PASS';
    html += `
      <div class="flow-step">
        <div class="flow-step-header">
          <span>Handoff #${idx+1}: ${h.source_agent} &rarr; ${h.target_agent}</span>
          <span class="badge ${isAlert ? 'badge mode' : 'badge online'}">${h.decision}</span>
        </div>
        <div><strong>Risk Score:</strong> ${(h.risk_score * 100).toFixed(1)}% (Threshold: ${(h.threshold * 100).toFixed(1)}%)</div>
        <div class="signal-tags">
          <span class="tag">Schema: ${(h.schema_score * 100).toFixed(1)}%</span>
          <span class="tag">Reasoning: ${(h.reasoning_score * 100).toFixed(1)}%</span>
          <span class="tag">NLI Consistency: ${(h.consistency_score * 100).toFixed(1)}%</span>
        </div>
        ${h.explanations && h.explanations.length ? `<div style="margin-top: 0.5rem; font-size: 0.85rem; color: #cbd5e1;"><em>${h.explanations[0]}</em></div>` : ''}
      </div>
    `;
  });

  html += `
    <div class="flow-step" style="border-color: #3b82f6;">
      <div class="flow-step-header">
        <span>Final Synthesizer Output</span>
        <span class="badge online">${data.error_contained ? 'ERROR CONTAINED' : 'COMPLETED'}</span>
      </div>
      <p style="font-size: 0.9rem; margin-top: 0.25rem;">${data.final_output ? data.final_output.answer : 'No output'}</p>
      <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">Total Latency: ${data.total_latency_ms.toFixed(1)} ms | Interceptor Overhead: ${data.interceptor_latency_ms.toFixed(1)} ms</div>
    </div>
  `;

  container.innerHTML = html;
}

async function loadCalibrationProfiles() {
  const container = document.getElementById('calibration-profiles-list');
  try {
    const res = await fetch('/calibration/deterministic-mock-v1');
    if (res.ok) {
      const data = await res.json();
      container.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    } else {
      container.innerHTML = '<pre>Profile not yet calibrated. Run `python -m app.cli calibrate`.</pre>';
    }
  } catch (e) {
    container.innerHTML = '<pre>Calibration profiles available in data/calibration_profiles/</pre>';
  }
}

async function submitAnnotation() {
  const rater = document.getElementById('ann-rater-id').value;
  const traceId = document.getElementById('ann-trace-id').value;
  const label = document.querySelector('input[name="ann-label"]:checked').value;
  const notes = document.getElementById('ann-notes').value;
  const statusEl = document.getElementById('ann-status');

  try {
    const res = await fetch('/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        trace_id: traceId,
        annotator_id: rater,
        label: label,
        notes: notes
      })
    });
    if (res.ok) {
      statusEl.innerText = "Annotation saved successfully!";
      statusEl.style.color = "#22c55e";
    }
  } catch (err) {
    statusEl.innerText = `Error: ${err.message}`;
    statusEl.style.color = "#ef4444";
  }
}
