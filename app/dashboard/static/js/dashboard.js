// Dashboard Interactive Logic & Real-Time API Telemetry

let activeCharts = {};
let currentOverviewFilter = 'live';
let currentTableFilter = 'all';
let cachedMetrics = null;

function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
  
  // Update sidebar active button
  const buttons = document.querySelectorAll('.nav-item');
  buttons.forEach(btn => {
    if (btn.getAttribute('onclick') && btn.getAttribute('onclick').includes(tabId)) {
      btn.classList.add('active');
    }
  });

  if (tabId === 'experiments') {
    loadTracesTable(currentTableFilter);
  } else if (tabId === 'model-comparison') {
    updateModelComparison();
  }
}

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  loadCalibrationProfiles();
  checkSystemHealth();
  loadOverviewMetrics('live');
  loadTracesTable('all');
});

async function checkSystemHealth() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const badge = document.getElementById('provider-badge');
    if (badge && data.default_provider) {
      badge.textContent = `Default: ${data.default_provider.toUpperCase()}`;
    }
  } catch (e) {
    console.error("Health check error:", e);
  }
}

function initCharts() {
  // Chart 1: ON vs OFF
  const ctxOnOff = document.getElementById('chart-on-off');
  if (ctxOnOff) {
    activeCharts.onOff = new Chart(ctxOnOff, {
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
    activeCharts.depth = new Chart(ctxDepth, {
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
    activeCharts.models = new Chart(ctxModels, {
      type: 'bar',
      data: {
        labels: ['deterministic-mock-v1', 'openai/gpt-oss-120b', 'gemini-flash-lite-latest'],
        datasets: [
          { label: 'Error Containment Rate (%)', data: [88.5, 85.0, 90.0], backgroundColor: '#22c55e' },
          { label: 'Unchecked Cascade Rate (%)', data: [11.5, 15.0, 10.0], backgroundColor: '#ef4444' }
        ]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true, max: 100 } } }
    });
  }

  // Chart 4: Architecture Comparison
  const ctxArch = document.getElementById('chart-architectures');
  if (ctxArch) {
    activeCharts.arch = new Chart(ctxArch, {
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
    activeCharts.calib = new Chart(ctxCalib, {
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
    activeCharts.ablation = new Chart(ctxAbl, {
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

// Set overview filter and update cards
function setOverviewFilter(filter) {
  currentOverviewFilter = filter;
  ['live', 'all', 'mock'].forEach(f => {
    const btn = document.getElementById(`btn-telemetry-${f}`);
    if (btn) btn.classList.toggle('active', f === filter);
  });

  const bannerText = document.getElementById('overview-source-text');
  if (bannerText) {
    if (filter === 'live') {
      bannerText.innerHTML = `<strong>Displaying Real-Time API Agent Telemetry</strong> (Captured from live Groq & Gemini multi-agent runs in SQLite)`;
    } else if (filter === 'all') {
      bannerText.innerHTML = `<strong>Displaying All Telemetry</strong> (Combined live API executions and deterministic mock validation traces)`;
    } else {
      bannerText.innerHTML = `<strong>Displaying Mock Benchmark Baseline</strong> (Controlled deterministic reproduction dataset)`;
    }
  }

  renderOverviewMetrics();
}

// Fetch metrics summary from API
async function loadOverviewMetrics(filter) {
  if (filter) currentOverviewFilter = filter;
  try {
    const res = await fetch('/metrics/summary');
    if (!res.ok) return;
    cachedMetrics = await res.json();
    renderOverviewMetrics();
    updateModelComparison();

    // Update status badge
    const apiBadge = document.getElementById('api-status-badge');
    if (apiBadge && cachedMetrics.providers_status) {
      const groqOk = cachedMetrics.providers_status.groq.configured;
      const geminiOk = cachedMetrics.providers_status.gemini.configured;
      if (groqOk && geminiOk) {
        apiBadge.innerHTML = "⚡ Groq & ✨ Gemini Online";
        apiBadge.style.background = "#064e3b";
        apiBadge.style.color = "#a7f3d0";
      } else if (groqOk || geminiOk) {
        apiBadge.innerHTML = `⚡ ${groqOk ? 'Groq' : 'Gemini'} Online`;
      }
    }
  } catch (err) {
    console.error("Failed to load metrics summary:", err);
  }
}

function renderOverviewMetrics() {
  if (!cachedMetrics) return;

  let stats = cachedMetrics.all;
  let subLabel = "Across all database traces";

  if (currentOverviewFilter === 'live') {
    stats = cachedMetrics.live;
    subLabel = "Live API LLM multi-agent executions";
  } else if (currentOverviewFilter === 'mock') {
    stats = cachedMetrics.mock;
    subLabel = "Offline deterministic benchmark traces";
  }

  const totalEl = document.getElementById('stat-total-traces');
  const totalSub = document.getElementById('stat-total-traces-sub');
  const containEl = document.getElementById('stat-containment-rate');
  const containSub = document.getElementById('stat-containment-sub');
  const cascadeEl = document.getElementById('stat-cascade-rate');
  const latencyEl = document.getElementById('stat-latency');
  const latencySub = document.getElementById('stat-latency-sub');

  if (totalEl) totalEl.textContent = stats.count || 0;
  if (totalSub) totalSub.textContent = subLabel;

  if (containEl) {
    if (stats.injected_count > 0) {
      containEl.textContent = `${stats.containment_rate}%`;
      containSub.textContent = `Contained ${Math.round((stats.containment_rate/100)*stats.injected_count)} of ${stats.injected_count} injected faults`;
    } else {
      containEl.textContent = "100%";
      containSub.textContent = "Clean execution (No faults active)";
    }
  }

  if (cascadeEl) {
    cascadeEl.textContent = `${stats.cascade_rate}%`;
  }

  if (latencyEl) {
    if (stats.avg_latency_ms >= 1000) {
      latencyEl.textContent = `${(stats.avg_latency_ms / 1000).toFixed(2)} s`;
      latencySub.textContent = "Real LLM multi-agent round-trip";
    } else {
      latencyEl.textContent = `${Math.round(stats.avg_latency_ms)} ms`;
      latencySub.textContent = "Offline execution latency";
    }
  }

  // Update Chart 1 if chart exists
  if (activeCharts.onOff && stats.injected_count > 0) {
    activeCharts.onOff.data.datasets[1].data = [stats.cascade_rate, stats.containment_rate];
    activeCharts.onOff.update();
  }
}

function updateModelComparison() {
  if (!cachedMetrics || !cachedMetrics.models) return;

  const tbody = document.getElementById('model-comparison-tbody');
  if (tbody) {
    let rows = '';
    const modelKeys = Object.keys(cachedMetrics.models);
    modelKeys.forEach(mName => {
      const m = cachedMetrics.models[mName];
      const isLive = (mName !== 'deterministic-mock-v1');
      const badge = isLive 
        ? `<span class="pill pill-live">⚡ REAL-TIME API</span>`
        : `<span class="pill pill-purple">🧪 MOCK</span>`;
      
      const latStr = m.avg_latency_ms >= 1000 
        ? `${(m.avg_latency_ms / 1000).toFixed(2)} s`
        : `${Math.round(m.avg_latency_ms)} ms`;

      rows += `
        <tr>
          <td><strong>${mName}</strong></td>
          <td>${badge}</td>
          <td>${m.count}</td>
          <td>${latStr}</td>
          <td class="text-success" style="font-weight: 600;">${m.containment_rate}%</td>
          <td class="text-danger">${m.cascade_rate}%</td>
        </tr>
      `;
    });

    if (rows) tbody.innerHTML = rows;
  }

  // Update Model Comparison Chart
  if (activeCharts.models) {
    const labels = Object.keys(cachedMetrics.models);
    const containData = labels.map(k => cachedMetrics.models[k].containment_rate);
    const cascadeData = labels.map(k => cachedMetrics.models[k].cascade_rate);

    activeCharts.models.data.labels = labels;
    activeCharts.models.data.datasets[0].data = containData;
    activeCharts.models.data.datasets[1].data = cascadeData;
    activeCharts.models.update();
  }
}

// Preset Prompts helper
function setPresetPrompt(text) {
  const input = document.getElementById('interactive-prompt');
  if (input) {
    input.value = text;
    input.focus();
  }
}

// Live Task Execution
async function runLiveTask() {
  const prompt = document.getElementById('interactive-prompt').value;
  const arch = document.getElementById('interactive-arch').value;
  const modelSelect = document.getElementById('interactive-model');
  const model = modelSelect ? modelSelect.value : 'groq';
  const fault = document.getElementById('interactive-fault').checked;
  const container = document.getElementById('live-trace-results');

  const modelLabel = model === 'groq' 
    ? 'GROQ (openai/gpt-oss-120b - REAL-TIME API)' 
    : model === 'gemini' 
    ? 'GOOGLE GEMINI (gemini-flash-lite-latest - REAL-TIME API)'
    : 'OFFLINE MOCK (Deterministic)';

  container.innerHTML = `
    <div class="flow-placeholder">
      <div style="margin-bottom: 0.5rem;"><span class="pulse-dot"></span> <strong>Executing Multi-Agent Pipeline...</strong></div>
      <div style="font-size: 0.88rem; color: #94a3b8;">Connecting to [${modelLabel}]...</div>
      <div style="font-size: 0.8rem; color: #64748b; margin-top: 0.4rem;">Coordinating Researcher &rarr; Verifier &rarr; Critic &rarr; Synthesizer with Runtime Interceptor validation.</div>
    </div>
  `;

  try {
    const res = await fetch('/run/task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: prompt,
        architecture: arch,
        model: model,
        interceptor_enabled: true,
        inject_fault: fault,
        fault_type: 'factual_mutation'
      })
    });
    
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned status ${res.status}`);
    }

    const data = await res.json();
    renderTraceFlow(data, container);

    // Refresh telemetry and table with the new live run
    loadOverviewMetrics();
    loadTracesTable(currentTableFilter);
  } catch (err) {
    container.innerHTML = `
      <div class="flow-placeholder text-danger" style="border-color: #ef4444;">
        <strong>Execution Failed:</strong> ${err.message}
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.5rem;">
          Check your API keys in .env or switch to the Offline Mock provider.
        </div>
      </div>
    `;
  }
}

function renderTraceFlow(data, container) {
  let html = '';

  const isLive = data.is_live_api || (data.model && data.model !== 'deterministic-mock-v1');
  const modelName = data.model || data.model_name || 'unknown';
  const providerName = (data.actual_provider || (isLive ? 'Real-Time API' : 'Mock')).toUpperCase();

  // 1. Task Prompt Hero Banner
  const liveBadge = isLive
    ? `<span class="pill pill-live">🟢 REAL-TIME API INFERENCE</span>`
    : `<span class="pill pill-blue">🧪 DETERMINISTIC MOCK</span>`;

  const faultBadge = data.fault_injected
    ? `<span class="pill pill-red">⚠️ Fault Injected (${data.fault_type || 'factual_mutation'})</span>`
    : `<span class="pill pill-green">✓ Clean Execution (No Fault)</span>`;

  const modelBadge = `<span class="pill pill-blue">🤖 Model: ${modelName}</span>`;
  const archBadge = `<span class="pill pill-purple">🏛️ Arch: ${data.architecture || 'sequential'}</span>`;
  const traceBadge = `<span class="pill pill-amber">🆔 Trace: ${data.trace_id || 'live'}</span>`;
  
  const fallbackNotice = data.fallback_used
    ? `<div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fde68a; padding: 0.5rem 0.75rem; border-radius: 0.375rem; margin-top: 0.5rem; font-size: 0.85rem;">
        ⚠️ <strong>Note:</strong> ${data.fallback_reason || 'Requested provider fell back to offline mock.'}
       </div>`
    : '';

  const latencySeconds = (data.total_latency_ms / 1000).toFixed(2);

  html += `
    <div class="prompt-hero">
      <div class="prompt-hero-header">
        <span>MULTI-AGENT TASK EXECUTION &bull; ${providerName}</span>
        <span>${new Date().toLocaleTimeString()} &bull; Total Latency: <strong>${data.total_latency_ms >= 1000 ? latencySeconds + 's' : Math.round(data.total_latency_ms) + 'ms'}</strong></span>
      </div>
      <div class="prompt-hero-text">"${data.prompt || 'No prompt specified'}"</div>
      <div class="prompt-badges">
        ${liveBadge}
        ${modelBadge}
        ${archBadge}
        ${faultBadge}
        ${traceBadge}
      </div>
      ${fallbackNotice}
    </div>
  `;

  // 2. Intermediate Agent Handoffs & Inspections
  const handoffs = data.handoffs || [];
  const agentOutputs = data.agent_outputs || {};

  const agentIcons = {
    researcher: '🔬',
    researcher_a: '🔬',
    researcher_b: '🔬',
    researcher_c: '🔬',
    verifier: '⚖️',
    critic: '🧐',
    supervisor: '👑',
    specialist: '🎯',
    synthesizer: '📑'
  };

  if (handoffs.length === 0) {
    html += '<div class="flow-step">Pipeline completed without intermediate handoffs.</div>';
  }

  handoffs.forEach((h, idx) => {
    const isAlert = h.decision === 'BLOCK' || h.decision === 'REPAIR';
    const isWarn = h.decision === 'PASS_WITH_WARNING';
    const boxClass = isAlert ? 'interceptor-box alert' : isWarn ? 'interceptor-box warning' : 'interceptor-box clean';
    const decStyle = isAlert ? 'background:#ef4444; color:#fff;' : isWarn ? 'background:#f59e0b; color:#fff;' : 'background:#22c55e; color:#fff;';
    
    const sourceIcon = agentIcons[h.source_agent] || '🤖';
    const targetIcon = agentIcons[h.target_agent] || '🤖';
    const srcOutput = agentOutputs[h.source_agent] || {};

    const riskPercent = Math.min(100, Math.round(h.risk_score * 100));
    const thPercent = Math.round(h.threshold * 100);
    const meterColor = isAlert ? '#ef4444' : isWarn ? '#f59e0b' : '#22c55e';

    html += `
      <div class="flow-step" style="border-left: 4px solid ${meterColor};">
        <!-- Step Header -->
        <div class="flow-step-header">
          <span>${sourceIcon} <strong>${h.source_agent.toUpperCase()}</strong> &rarr; ${targetIcon} <strong>${h.target_agent.toUpperCase()}</strong> (Handoff #${idx+1})</span>
          <span class="badge" style="${decStyle}">${h.decision}</span>
        </div>

        <!-- Source Agent's Generated Answer -->
        ${srcOutput.answer ? `
          <div style="margin-top: 0.5rem;">
            <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Observable Output from ${h.source_agent}:</span>
            <div class="agent-output-box">
              ${srcOutput.answer}
            </div>
          </div>
        ` : ''}

        <!-- Observable Claims List (if any) -->
        ${srcOutput.claims && srcOutput.claims.length ? `
          <div class="claims-container">
            <span style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Claims & Evidence Grounding:</span>
            ${srcOutput.claims.map(c => `
              <div class="claim-card">
                <div><strong>Claim:</strong> "${c.claim}" <span class="pill pill-blue" style="font-size: 0.7rem; padding: 0.1rem 0.4rem;">${Math.round(c.confidence * 100)}% Conf</span></div>
                ${c.evidence && c.evidence.length ? `<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem;">↳ <em>Evidence: ${c.evidence.join('; ')}</em></div>` : ''}
              </div>
            `).join('')}
          </div>
        ` : ''}

        <!-- Runtime Interceptor Diagnostic Box -->
        <div class="${boxClass}">
          <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; font-weight: 600;">
            <span>🛡️ Runtime Interceptor Diagnostic Evaluation</span>
            <span style="color: ${meterColor};">Risk: ${riskPercent}% (Threshold: ${thPercent}%)</span>
          </div>

          <div class="risk-meter">
            <div class="risk-fill" style="width: ${riskPercent}%; background: ${meterColor};"></div>
          </div>

          <div class="signal-tags">
            <span class="tag">Signal 1 (Schema): ${Math.round(h.schema_score * 100)}%</span>
            <span class="tag">Signal 2 (Reasoning): ${Math.round(h.reasoning_score * 100)}%</span>
            <span class="tag">Signal 3 (NLI Consistency): ${Math.round(h.consistency_score * 100)}%</span>
            <span class="tag" style="background: #1e293b;">Inspection Latency: ${(h.latency_ms || 0).toFixed(1)}ms</span>
          </div>

          ${h.explanations && h.explanations.length ? `
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #cbd5e1; border-top: 1px solid #334155; padding-top: 0.4rem;">
              <strong>Diagnostic Explanations:</strong>
              <ul style="margin-left: 1.2rem; margin-top: 0.2rem;">
                ${h.explanations.map(e => `<li>${e}</li>`).join('')}
              </ul>
            </div>
          ` : ''}
        </div>
      </div>
    `;
  });

  // 3. Final Synthesizer Hero Card
  const contained = data.error_contained;
  const finalClass = contained ? 'final-hero' : 'final-hero uncontained';
  const finalBadgeStyle = contained ? 'background: #059669; color: white;' : 'background: #dc2626; color: white;';
  const finalBadgeText = contained ? '🛡️ CASCADE CONTAINED' : '⚠️ UNCHECKED CASCADE';

  let consolidatedAnswer = 'No output produced';
  if (data.final_output) {
    if (typeof data.final_output === 'string') {
      consolidatedAnswer = data.final_output;
    } else if (data.final_output.answer) {
      consolidatedAnswer = data.final_output.answer;
    } else {
      consolidatedAnswer = JSON.stringify(data.final_output, null, 2);
    }
  }

  html += `
    <div class="${finalClass}">
      <div class="flow-step-header" style="align-items: center;">
        <span style="font-size: 1.1rem; font-weight: 700;">📑 Final Synthesizer Output</span>
        <span class="badge" style="${finalBadgeStyle} font-size: 0.85rem; padding: 0.4rem 0.8rem;">${finalBadgeText}</span>
      </div>
      <div class="agent-output-box" style="background: rgba(15, 23, 42, 0.85); border-color: rgba(255, 255, 255, 0.15); font-size: 1rem; margin-top: 0.75rem; line-height: 1.6;">
        <strong>Consolidated Answer:</strong><br>
        ${consolidatedAnswer}
      </div>
      <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 0.75rem; display: flex; gap: 1.2rem; flex-wrap: wrap;">
        <span><strong>Total Latency:</strong> ${data.total_latency_ms >= 1000 ? (data.total_latency_ms/1000).toFixed(2) + ' s' : Math.round(data.total_latency_ms) + ' ms'}</span>
        <span><strong>Interceptor Overhead:</strong> ${(data.interceptor_latency_ms || 0).toFixed(1)} ms</span>
        <span><strong>Model:</strong> ${modelName}</span>
        <span><strong>Cascade Depth:</strong> ${data.cascade_depth || 0} agents</span>
      </div>
    </div>
  `;

  container.innerHTML = html;
}

// Traces Explorer Table
async function loadTracesTable(filter) {
  if (filter) currentTableFilter = filter;
  ['all', 'realtime', 'mock'].forEach(f => {
    const btn = document.getElementById(`tbl-filter-${f}`);
    if (btn) btn.classList.toggle('active', f === currentTableFilter);
  });

  const box = document.getElementById('traces-table-box');
  if (!box) return;

  box.innerHTML = `<p style="color: #94a3b8;"><span class="pulse-dot"></span> Loading traces from SQLite database...</p>`;

  try {
    const res = await fetch(`/traces?limit=100&model_filter=${currentTableFilter}`);
    if (!res.ok) throw new Error("Failed to load traces");
    const traces = await res.json();

    if (!traces || traces.length === 0) {
      box.innerHTML = `<p style="color: #94a3b8; padding: 1rem;">No traces found for filter '${currentTableFilter}'. Execute a task in the Live Trace Viewer to see real-time agent results!</p>`;
      return;
    }

    let rows = '';
    traces.forEach((t, i) => {
      const isLive = t.is_live_api;
      const typeBadge = isLive 
        ? `<span class="pill pill-live">⚡ REAL-TIME</span>`
        : `<span class="pill pill-purple">🧪 MOCK</span>`;

      const containBadge = t.error_contained 
        ? `<span class="pill pill-green">🛡️ Contained</span>`
        : `<span class="pill pill-red">⚠️ Cascade</span>`;

      const latStr = t.total_latency_ms >= 1000 
        ? `${(t.total_latency_ms / 1000).toFixed(2)}s`
        : `${Math.round(t.total_latency_ms)}ms`;

      const promptDisplay = t.prompt 
        ? (t.prompt.length > 50 ? t.prompt.substring(0, 50) + '...' : t.prompt)
        : 'Task execution trace';

      rows += `
        <tr>
          <td>${i+1}</td>
          <td><code>${t.trace_id}</code></td>
          <td>${typeBadge}</td>
          <td><strong>${t.model}</strong></td>
          <td>${t.architecture}</td>
          <td title="${t.prompt || ''}">${promptDisplay}</td>
          <td>${containBadge}</td>
          <td>${latStr}</td>
          <td>
            <button class="table-btn" onclick="inspectTraceById('${t.trace_id}')">🔍 Inspect</button>
          </td>
        </tr>
      `;
    });

    box.innerHTML = `
      <table class="data-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Trace ID</th>
            <th>Type</th>
            <th>Model Provider</th>
            <th>Architecture</th>
            <th>Task Prompt</th>
            <th>Containment</th>
            <th>Latency</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
    `;
  } catch (err) {
    box.innerHTML = `<p class="text-danger">Failed to load traces: ${err.message}</p>`;
  }
}

// Inspect past trace by ID
async function inspectTraceById(traceId) {
  switchTab('trace-viewer');
  const container = document.getElementById('live-trace-results');
  container.innerHTML = `<div class="flow-placeholder"><span class="pulse-dot"></span> Loading trace [${traceId}] from SQLite...</div>`;
  container.scrollIntoView({ behavior: 'smooth' });

  try {
    const res = await fetch(`/traces/${traceId}`);
    if (!res.ok) throw new Error("Trace not found");
    const data = await res.json();
    renderTraceFlow(data, container);
  } catch (err) {
    container.innerHTML = `<div class="flow-placeholder text-danger">Failed to load trace: ${err.message}</div>`;
  }
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
