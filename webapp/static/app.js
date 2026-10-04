/* CISCaRL Live — frontend logic.
 * Tries the FastAPI backend first; falls back to static data.json
 * (so the GitHub Pages site still shows rules + accuracy without a server). */

const API = (window.CISARL_API || "").replace(/\/$/, "");
const METHOD_ORDER = ["CISCaRL", "SCRE", "Cox T-learner", "CSF (black-box)",
                      "Bo & Ding", "Hybrid", "CRE"];

let META = null, RULES = null, METRICS = null, OFFLINE = false;
let barChart = null, radarChart = null;
let activeTab = "CISCaRL";

async function apiGet(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}
async function apiPost(path, body) {
  const res = await fetch(API + path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

function el(id) { return document.getElementById(id); }
function fmt(x, d = 4) { return (x === null || x === undefined || Number.isNaN(x)) ? "n/a" : Number(x).toFixed(d); }

async function boot() {
  try {
    META = await apiGet("/api/meta");
    RULES = await apiGet("/api/rules");
    METRICS = await apiGet("/api/metrics?regime=original");
    el("apiStatus").textContent = "● live backend connected";
    el("apiStatus").style.color = "#22c55e";
  } catch (e) {
    try {
      const d = await (await fetch("data.json")).json();
      META = d.meta; RULES = d.rules; METRICS = d.metrics;
      window.__DEMO = d.demo_prediction;
      OFFLINE = true;
      el("apiStatus").textContent = "◐ offline preview (static data) — run the backend for live inputs";
      el("apiStatus").style.color = "#f59e0b";
    } catch (e2) {
      el("apiStatus").textContent = "backend unreachable";
      el("apiStatus").style.color = "#ef4444";
      return;
    }
  }
  buildForm();
  fillDemo();
  renderRules();
  renderMetrics();
}

function buildForm() {
  const f = el("form");
  f.innerHTML = "";
  META.feature_names.forEach(name => {
    const r = META.ranges[name] || {};
    const div = document.createElement("div");
    div.className = "field";
    div.innerHTML =
      `<label>${name}</label>
       <input id="in_${name}" type="number" step="any" value="${r.median ?? 0}">
       <span class="rng">${r.min !== undefined ? r.min.toFixed(2) + " – " + r.max.toFixed(2) : ""}</span>`;
    f.appendChild(div);
  });
}

function fillDemo() {
  const d = META.demo_patient;
  META.feature_names.forEach(name => {
    const inp = el("in_" + name);
    if (inp) inp.value = d[name];
  });
}

function readPatient() {
  const p = {};
  META.feature_names.forEach(name => {
    const v = parseFloat(el("in_" + name).value);
    p[name] = Number.isNaN(v) ? META.ranges[name].median : v;
  });
  return p;
}

async function run() {
  const btn = el("btnRun");
  btn.classList.add("loading"); btn.textContent = "running…";
  const patient = readPatient();
  let result = null;
  if (!OFFLINE) {
    try { result = await apiPost("/api/predict", { patient }); } catch (e) { result = null; }
  }
  if (!result || !result.predictions) {
    result = window.__DEMO || (await (await fetch("data.json")).json()).demo_prediction;
  }
  result.t_star = result.t_star || META.t_star;
  result.warnings = result.warnings || [];
  window.__LAST_MATCH = result.cis_matched_rule;
  renderCIS(result.cis_matched_rule, result.t_star, result.warnings);
  renderPredictions(result.predictions);
  renderRules();
  btn.classList.remove("loading"); btn.textContent = "▶ Run all methods";
}

function recoClass(s) {
  if (!s) return "sugg";
  if (s.includes("Treat")) return "treat";
  if (s.includes("Avoid")) return "avoid";
  return "sugg";
}

function renderCIS(mr, tStar, warnings) {
  const box = el("cisResult");
  if (!mr) { box.innerHTML = '<div class="note">No result.</div>'; return; }
  const cls = recoClass(mr.recommendation);
  // confidence-interval bar
  const lo = mr.ci_low, hi = mr.ci_high;
  const pad = (hi - lo) * 0.15 + 0.02;
  const loR = Math.min(lo - pad, -0.02), hiR = Math.max(hi + pad, 0.02);
  const span = hiR - loR;
  const leftPct = ((lo - loR) / span) * 100;
  const widthPct = ((hi - lo) / span) * 100;
  const zeroPct = ((0 - loR) / span) * 100;
  box.innerHTML = `
    <div class="reco ${cls}">${mr.recommendation}</div>
    <div class="cis-rule">IF ${mr.condition_str}</div>
    <div class="cis-stats">
      <div><span>Group effect</span><b>${fmt(mr.mean_cate)}</b></div>
      <div><span>90% CI</span><b>[${fmt(mr.ci_low, 3)}, ${fmt(mr.ci_high, 3)}]</b></div>
      <div><span>Stability</span><b>${mr.stability === null ? "—" : (mr.stability * 100).toFixed(0) + "%"}</b></div>
      <div><span>Support</span><b>${mr.support}</b></div>
      <div><span>Horizon t*</span><b>${Math.round(tStar)} d</b></div>
    </div>
    <div class="ci-bar" title="90% conformal interval">
      <div class="ci-fill" style="left:${leftPct}%;width:${widthPct}%"></div>
      <div class="ci-zero" style="left:${zeroPct}%"></div>
    </div>
    ${(warnings && warnings.length) ? `<div class="note" style="color:#fca5a5">Out of range: ${warnings.join("; ")}</div>` : ""}
    <div class="note">Group-level effect (subgroup mean), not an individual prediction.</div>`;
}

function renderPredictions(preds) {
  if (!preds) return;
  const names = METHOD_ORDER.filter(n => n in preds);
  const vals = names.map(n => preds[n]);
  const finite = vals.filter(v => v !== null && v !== undefined);
  const maxAbs = Math.max(0.01, ...finite.map(v => Math.abs(v)));

  // cards
  const cardBox = el("predCards");
  cardBox.innerHTML = "";
  names.forEach(n => {
    const v = preds[n];
    const pct = v === null ? 0 : Math.min(100, Math.abs(v) / maxAbs * 100);
    const col = v === null ? "#666" : v >= 0 ? "#22c55e" : "#ef4444";
    cardBox.innerHTML += `
      <div class="pred-card">
        <div class="m">${n}</div>
        <div class="v" style="color:${col}">${v === null ? "n/a" : (v >= 0 ? "+" : "") + v.toFixed(3)}</div>
        <div class="bar"><i style="width:${pct}%;background:${col}"></i></div>
      </div>`;
  });

  // bar chart
  const ctx = el("barChart");
  if (barChart) barChart.destroy();
  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: names,
      datasets: [{
        label: "Predicted CATE @ t*",
        data: vals.map(v => v === null ? 0 : v),
        backgroundColor: vals.map(v => v === null ? "#475569" : v >= 0 ? "rgba(34,197,94,.75)" : "rgba(239,68,68,.75)"),
        borderColor: vals.map(v => v === null ? "#475569" : v >= 0 ? "#22c55e" : "#ef4444"),
        borderWidth: 1.5, borderRadius: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#9aa7c7", font: { size: 10 } }, grid: { color: "rgba(255,255,255,.06)" } },
        y: { ticks: { color: "#9aa7c7" }, grid: { color: "rgba(255,255,255,.06)" } }
      }
    }
  });
}

function renderRules() {
  const tabs = el("ruleTabs");
  tabs.innerHTML = "";
  const names = Object.keys(RULES);
  // CISCaRL first
  names.sort((a, b) => (a === "CISCaRL" ? -1 : b === "CISCaRL" ? 1 : 0));
  names.forEach(n => {
    const info = RULES[n];
    const b = document.createElement("button");
    b.className = "tab" + (n === activeTab ? " active" : "");
    b.innerHTML = `${n}<span class="cnt">${info.count}</span>`;
    b.onclick = () => { activeTab = n; renderRules(); };
    tabs.appendChild(b);
  });
  const info = RULES[activeTab];
  const body = el("ruleBody");
  body.innerHTML = "";
  if (!info || info.count === 0) {
    body.innerHTML = `<div class="note">${info && info.note ? info.note : "No rules."}</div>`;
    return;
  }
  if (info.shown && info.count > info.shown) {
    body.innerHTML = `<div class="note">Showing top ${info.shown} of <b>${info.count}</b> rules by |coefficient|.</div>`;
  }
  const matchedCond = (window.__LAST_MATCH && window.__LAST_MATCH.condition_str) || null;
  info.rules.forEach(r => {
    const isMatched = matchedCond && r.condition_str === matchedCond;
    const div = document.createElement("div");
    div.className = "rule" + (isMatched ? " matched" : "");
    let right;
    if (info.kind === "rich") {
      const rec = r.recommendation.startsWith("HIGH CONFIDENCE") ? r.recommendation.split(":")[1].trim() : "suggestive";
      right = `<div class="val">${r.mean_cate >= 0 ? "+" : ""}${r.mean_cate.toFixed(3)}<div class="meta">${(r.stability * 100).toFixed(0)}% · n=${r.support}</div></div>`;
      div.innerHTML = `<div class="idx">${r.index}</div>
        <div><div class="cond">${r.condition_str}</div>
        <div class="meta">90% CI [${r.ci_low.toFixed(3)}, ${r.ci_high.toFixed(3)}] · ${rec}</div></div>${right}`;
    } else {
      right = `<div class="val">${r.coef >= 0 ? "+" : ""}${r.coef.toFixed(3)}<div class="meta">coef</div></div>`;
      div.innerHTML = `<div class="idx">${r.index}</div>
        <div><div class="cond">${r.condition_str}</div>
        <div class="meta">weighted rule term</div></div>${right}`;
    }
    body.appendChild(div);
  });
}

function renderMetrics() {
  const m = METRICS;
  if (!m || !m.available) { el("metricTable").innerHTML = "<tr><td>No metrics.</td></tr>"; return; }
  const methods = m.methods;
  const cols = m.metrics;
  let html = "<thead><tr><th style='text-align:left'>Method</th>" +
    cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
  // order rows by MAE
  const ordered = [...methods].sort((a, b) => a.MAE - b.MAE);
  ordered.forEach(row => {
    html += `<tr><td class="method">${row.method}</td>`;
    cols.forEach(c => {
      const r = row["rank_" + c];
      const cls = r === 1 ? "rank1" : "";
      const dec = c === "Rules" ? 1 : (c === "R2" ? 2 : 3);
      html += `<td class="${cls}">${fmt(row[c], dec)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";
  el("metricTable").innerHTML = html;

  // radar: normalized accuracy profile (MAE, RMSE inverted; Spearman, F1, AUC)
  const radarMetrics = ["MAE", "RMSE", "Spearman", "F1", "AUC"];
  const norm = {};
  radarMetrics.forEach(met => {
    const vals = methods.map(r => r[met]);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    norm[met] = methods.map(r => {
      const t = hi === lo ? 1 : (r[met] - lo) / (hi - lo);
      return (met === "MAE" || met === "RMSE") ? 1 - t : t; // higher = better
    });
  });
  const palette = ["#22c55e", "#4f7cff", "#a855f7", "#f59e0b", "#22d3ee", "#ec4899", "#94a3b8", "#f472b6", "#84cc16"];
  const ctx = el("radarChart");
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: radarMetrics.map(x => x + (x === "MAE" || x === "RMSE" ? " (inv)" : "")),
      datasets: methods.map((r, i) => ({
        label: r.method,
        data: radarMetrics.map(met => norm[met][methods.indexOf(r)]),
        borderColor: palette[i % palette.length],
        backgroundColor: palette[i % palette.length] + "22",
        borderWidth: 2, pointRadius: 2,
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#cbd5e1", font: { size: 10 } } } },
      scales: { r: {
        angleLines: { color: "rgba(255,255,255,.1)" },
        grid: { color: "rgba(255,255,255,.1)" },
        pointLabels: { color: "#9aa7c7", font: { size: 11 } },
        ticks: { display: false }, min: 0, max: 1
      } }
    }
  });
}

// keep the matched rule highlighted when rules re-render
el("btnDemo").onclick = fillDemo;
el("btnRun").onclick = run;
boot();
