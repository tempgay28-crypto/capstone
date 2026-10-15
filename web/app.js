"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const form = $("#form");
const report = $("#report");
const errorBox = $("#error");
const EMPTY = report.innerHTML;
let META = null;
let LAST = null;
let sdTouched = false;

const fmt = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(Number(v))
    ? "—"
    : Number(v).toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });

const VERDICTS = {
  compliant: ["v-good", "✓", "Compliant"],
  deficient: ["v-bad", "✕", "Deficient"],
  exceeded: ["v-bad", "✕", "Exceeded"],
  inadequate: ["v-bad", "✕", "Inadequate"],
  marginal: ["v-warn", "!", "Marginal"],
  informative: ["v-info", "i", "Informative"],
  "not checked": ["v-none", "–", "Not checked"],
};

// ------------------------------------------------------------------ setup

async function init() {
  const theme = new URLSearchParams(location.search).get("theme");  // ?theme=light or ?theme=dark
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  try {
    META = await (await fetch("/api/meta")).json();
  } catch {
    showError("Could not reach the assessment server. Is app.py running?");
    return;
  }
  const exp = form.elements.exposure_class;
  META.exposure_classes.forEach((c) => exp.add(new Option(c, c)));
  for (const sel of form.querySelectorAll("select[data-choices]")) {
    (META.choices[sel.name] || []).forEach((c) => sel.add(new Option(c, c)));
  }
  const cm = form.elements.corr_method;
  Object.entries(META.corrosion_methods).forEach(([k, m]) => cm.add(new Option(`${m.label} (±${m.sd})`, k)));
  const sd = form.elements.corr_sd_pp;
  sd.placeholder = `${META.corrosion_methods.judgement.sd} if not set`;
  cm.addEventListener("change", () => {
    if (sdTouched) return;
    const m = META.corrosion_methods[cm.value];
    sd.value = m ? m.sd : "";
  });
  sd.addEventListener("input", () => { sdTouched = sd.value.trim() !== ""; });
  const ex = new URLSearchParams(location.search).get("example");  // ?example=inspection or ?example=measured
  if (ex !== null) {
    fill(META.examples[ex] || META.examples.inspection);
    form.requestSubmit();
  }
}

function values() {
  const out = {};
  for (const el of form.elements) if (el.name) out[el.name] = el.value.trim();
  return out;
}

function fill(obj) {
  form.reset();
  for (const [k, v] of Object.entries(obj)) if (form.elements[k]) form.elements[k].value = v;
  sdTouched = true;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  form.querySelectorAll("[required]").forEach((el) => el.classList.toggle("invalid", !el.value.trim()));
  const btn = form.querySelector("button.primary");
  btn.disabled = true;
  try {
    const res = await fetch("/api/assess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Assessment failed.");
    LAST = data;
    render(data);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
  }
});

form.querySelectorAll("[data-example]").forEach((btn) =>
  btn.addEventListener("click", () => {
    if (!META) return;
    fill(META.examples[btn.dataset.example]);
    form.requestSubmit();
  }));

$("#clear").addEventListener("click", () => {
  form.reset();
  sdTouched = false;
  LAST = null;
  errorBox.hidden = true;
  form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
  report.innerHTML = EMPTY;
});

// ------------------------------------------------------------------ report

function el(tag, text, cls) {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
}

function badge(verdict) {
  const [cls, icon, label] = VERDICTS[verdict] || VERDICTS["not checked"];
  const b = el("span", undefined, `badge ${cls}`);
  b.append(el("span", icon, "ic"), document.createTextNode(label));
  return b;
}

function render(d) {
  const node = $("#report-tpl").content.cloneNode(true);
  const f = (k) => node.querySelector(`[data-f="${k}"]`);
  const fl = d.flexure;
  const sh = d.shear;

  f("r-label").textContent = fl.corrosion_source === "inferred"
    ? "Flexure — learned model · corrosion inferred" : "Flexure — learned model · corrosion measured";
  f("r").textContent = `${fmt(fl.residual_pct)} %`;
  f("r-range").textContent = `90 % range ${fmt(fl.low_pct)}–${fmt(fl.high_pct)} %`;
  f("r-knm").textContent = `≈ ${fmt(fl.mu_res_knm)} kN·m of ${fmt(fl.mu0_knm)} kN·m (IS 456 design)`;
  f("ana").textContent = `${fmt(fl.area_loss_residual_pct)} %`;
  f("ana-knm").textContent = `${fmt(fl.mu_area_loss_knm)} kN·m · steel-area loss only`;
  f("v").textContent = `${fmt(sh.after.Vu_kn, 0)} kN`;
  f("v-meta").textContent = `of ${fmt(sh.before.Vu_kn, 0)} kN uncorroded · ${fmt(sh.retained_pct, 0)} % retained`;
  f("comp").textContent = `${fmt(d.completeness.pct, 0)} %`;
  f("comp-meta").textContent = `${d.completeness.given} of ${d.completeness.of} optional inputs`;
  f("rangebar").append(rangeBar(fl));

  const unc = f("unc");
  fl.uncertainty.forEach((u) => {
    const tr = el("tr");
    tr.append(el("td", u.source), el("td", fmt(u.sd), "num"), el("td", u.what));
    unc.append(tr);
  });
  const total = el("tr", undefined, "total");
  total.append(el("td", "Combined"), el("td", fmt(fl.total_sd), "num"),
    el("td", `90 % range = estimate ± 1.645 × ${fmt(fl.total_sd)}, limited to 0–100 %`));
  unc.append(total);

  f("curve").append(curveChart(fl, d.inputs.corr_pct));

  d.cautions.forEach((c) => f("cautions").append(el("li", c)));

  const checks = f("checks");
  d.checks.forEach((c) => {
    const tr = el("tr");
    const check = el("td", c.check);
    if (c.note) check.append(el("span", c.note, "note"));
    const verdict = el("td");
    verdict.append(badge(c.verdict));
    tr.append(el("td", c.clause), check, el("td", c.required), el("td", c.actual), verdict);
    checks.append(tr);
  });

  const p = d.provenance;
  f("prov").textContent =
    `Learned model: ensemble of 5 neural networks (M3) trained on ${p.trained_rows} beams from published tests. ` +
    `On ${p.rows_test} beams from ${p.papers_test} papers it never saw: R² ${fmt(p.test_r2, 2)}, RMSE ${fmt(p.test_rmse)} ` +
    `points (${fmt(p.rmse_corroded)} for corroded beams). Specimen scatter ${fmt(p.replicate_sd)} points, from ` +
    `${p.replicate_specimens} replicate specimens. IS 456 values use the design formulae (0.87 fy, Tables 19 and 20) ` +
    `with corrosion as loss of steel area only.`;

  renderStage2(d, f);
  report.replaceChildren(node);
  report.querySelector('[data-act="json"]').addEventListener("click", downloadJson);
  report.querySelector('[data-act="print"]').addEventListener("click", () => window.print());
}

function renderStage2(d, f) {
  const x = d.inputs;
  const c = d.condition;
  const corr = d.corrosion;

  // Identification line (only what was entered)
  const ident = f("ident");
  const bits = [x.member_ref, x.structure, x.location].filter(Boolean);
  if (bits.length) {
    ident.append(el("b", bits[0]));
    bits.slice(1).forEach((b) => ident.append(document.createTextNode(` · ${b}`)));
  }
  if (x.inspector || x.inspection_date) {
    const by = [x.inspector ? `inspected by ${x.inspector}` : "inspected", x.inspection_date ? `on ${x.inspection_date}` : ""];
    ident.append(document.createTextNode(`${ident.childNodes.length ? " · " : ""}${by.join(" ").trim()}`));
  }

  // Corrosion state
  f("state").textContent = c.state;
  f("state-meta").textContent = `probability ${fmt(c.state_prob, 2)} · ` + (corr.source === "measured"
    ? `measured mass loss ${fmt(corr.mean)} %` : `expected mass loss ${fmt(c.mass_loss_mean)} %`);
  const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
  const info = [
    ["90 % range of mass loss (state model)", `${fmt(c.mass_loss_90[0])}–${fmt(c.mass_loss_90[1])} %`],
    ["Corrosion level used", `${corr.source} ${fmt(corr.mean)} % ± ${fmt(corr.sd)}`],
    ["Remaining uncertainty", `${fmt(c.entropy_bits, 2)} bits (prior ${fmt(c.prior_entropy_bits, 2)})`],
    ["Information from instruments", `${fmt(c.info_instruments_bits, 2)} bits · ${plural(c.n_instruments, "reading")}`],
    ["Information from observations", `${fmt(c.info_observations_bits, 2)} bits · ${plural(c.n_observations, "field")}`],
  ];
  if (c.n_measured) info.push(["Information from measurement", `${fmt(c.info_measurement_bits, 2)} bits`]);
  info.forEach(([k, v]) => {
    const tr = el("tr");
    tr.append(el("td", k), el("td", v));
    f("state-info").append(tr);
  });
  const bars = f("state-bars");
  c.states.forEach((name, i) => {
    const lab = el("span", `${name} (${c.state_bins[i][0]}–${c.state_bins[i][1]} %)`, "name");
    if (name === c.state) lab.classList.add("best");
    const track = el("div", undefined, "track");
    const fill = el("div", undefined, "fill");
    fill.style.width = `${(c.posterior[i] * 100).toFixed(1)}%`;
    const tick = el("div", undefined, "prior");
    tick.style.left = `calc(${(c.prior[i] * 100).toFixed(1)}% - 1px)`;
    tick.title = `Prior ${fmt(c.prior[i] * 100, 0)} %`;
    track.append(fill, tick);
    track.title = `${name}: ${fmt(c.posterior[i] * 100, 1)} % (prior ${fmt(c.prior[i] * 100, 1)} %)`;
    bars.append(lab, track, el("span", `${fmt(c.posterior[i] * 100, 0)} %`, "pct"));
  });
  bars.append(el("div", "Bar: probability with all evidence · tick: prior from exposure class and age", "bars-legend"));

  f("ev-count").textContent = `(${c.evidence_used.length})`;
  if (!c.evidence_used.length) {
    const tr = el("tr");
    const td = el("td", "No NDT or visual evidence entered.");
    td.colSpan = 5;
    tr.append(td);
    f("evidence").append(tr);
  }
  c.evidence_used.forEach((e) => {
    const tr = el("tr");
    tr.append(el("td", e.evidence), el("td", String(e.value)), el("td", e.band), el("td", e.favours), el("td", e.source));
    f("evidence").append(tr);
  });
  const useful = c.next_observations.filter((n) => n.gain_bits >= 0.01);
  useful.forEach((n) => {
    const tr = el("tr");
    tr.append(el("td", n.test), el("td", n.kind), el("td", n.cost), el("td", fmt(n.gain_bits, 2), "num"));
    f("next").append(tr);
  });
  if (!useful.length) {
    const tr = el("tr");
    const td = el("td", "Further tests would add little: the corrosion state is already well determined.");
    td.colSpan = 4;
    tr.append(td);
    f("next").append(tr);
  }

  // Rehabilitation strategy
  const r = d.rehabilitation;
  const u = r.urgency;
  const banner = el("div", undefined, `urgency u-${u.level.toLowerCase()}`);
  const body = el("div");
  body.append(el("div", `Urgency: ${u.level} — ${u.action}`, "title"));
  const ul = el("ul");
  u.structural.forEach((t) => ul.append(el("li", `Structural: ${t}`)));
  u.durability.forEach((t) => ul.append(el("li", `Durability: ${t}`)));
  body.append(ul);
  banner.append(el("span", u.level === "Routine" ? "✓" : "!", "ic"), body);
  f("urgency").append(banner);

  const groups = {};
  r.interventions.forEach((it) => (groups[it.group] ||= []).push(it));
  Object.entries(groups).forEach(([g, items]) => {
    const box = el("div", undefined, "group");
    box.append(el("h3", g));
    items.forEach((it) => {
      const card = el("div", undefined, "iv");
      const head = el("div", undefined, "iv-head");
      head.append(el("span", it.role, `role role-${it.role.toLowerCase()}`), el("span", it.name, "iv-name"));
      card.append(head, el("div", it.principle, "principle"), el("p", it.purpose));
      const trig = el("div", undefined, "trig");
      trig.append(el("b", "Triggered by: "), document.createTextNode(it.triggered_by.join("; ")));
      card.append(trig);
      if (it.limitations.length) {
        const det = el("details");
        det.append(el("summary", `Limitations (${it.limitations.length})`));
        const lu = el("ul");
        it.limitations.forEach((l) => lu.append(el("li", l)));
        det.append(lu);
        card.append(det);
      }
      card.append(el("div", `References: ${it.references.join(" · ")}`, "refs"));
      box.append(card);
    });
    f("interventions").append(box);
  });

  const spec = r.material_spec;
  if (!spec) {
    f("spec-panel").remove();
    return;
  }
  [["Exposure class (IS 456 Table 3)", spec.exposure_class], ["Minimum concrete grade", spec.min_grade],
   ["Maximum free w/c ratio", fmt(spec.max_wc, 2)], ["Minimum cement content", `${spec.min_cement_kg_m3} kg/m³`],
   ["Nominal cover to reinforcement", `${spec.nominal_cover_mm} mm`], ["Repair mortar", spec.repair_mortar]]
    .forEach(([k, v]) => {
      const tr = el("tr");
      tr.append(el("td", k), el("td", v));
      f("spec").append(tr);
    });
  f("spec").closest("table").after(el("p", spec.note, "muted small-note"));
}

function downloadJson() {
  if (!LAST) return;
  const blob = new Blob([JSON.stringify(LAST, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `rc-assessment-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ------------------------------------------------------------------ charts (inline SVG)

function legend(items) {
  const lg = el("div", undefined, "legend");
  items.forEach(([kind, color, text, extra = ""]) => {
    const s = el("span");
    const key = el("i", undefined, `key-${kind}`);
    key.style.cssText = `background:${color};${extra}`;
    s.append(key, document.createTextNode(text));
    lg.append(s);
  });
  return lg;
}

function rangeBar(fl) {
  const W = 640, x0 = 18, x1 = W - 18, max = 120, y = 30;
  const X = (v) => x0 + (Math.max(0, Math.min(max, v)) / max) * (x1 - x0);
  let s = `<svg viewBox="0 0 ${W} 66" role="img" aria-label="Residual flexural strength ${fmt(fl.residual_pct)} %, 90 % range ${fmt(fl.low_pct)} to ${fmt(fl.high_pct)} %">`;
  for (let v = 0; v <= max; v += 20) {
    s += `<line x1="${X(v)}" x2="${X(v)}" y1="20" y2="40" style="stroke:var(--grid)"/>`;
    s += `<text x="${X(v)}" y="58" text-anchor="middle" style="fill:var(--muted);font-size:11px">${v} %</text>`;
  }
  s += `<line x1="${x0}" x2="${x1}" y1="${y}" y2="${y}" style="stroke:var(--grid)"/>`;
  s += `<line x1="${X(100)}" x2="${X(100)}" y1="16" y2="44" style="stroke:var(--axis);stroke-width:1.5"/>`;
  if (fl.upper_truncated) {
    s += `<rect x="${X(100)}" y="${y - 7}" width="${X(fl.high_raw_pct) - X(100)}" height="14" rx="4"
      style="fill:none;stroke:var(--s1);stroke-dasharray:3 3;opacity:.7"/>`;
  }
  s += `<rect x="${X(fl.low_pct)}" y="${y - 7}" width="${Math.max(2, X(fl.high_pct) - X(fl.low_pct))}" height="14" rx="4" style="fill:var(--s1);opacity:.22"/>`;
  s += `<line x1="${X(fl.area_loss_residual_pct)}" x2="${X(fl.area_loss_residual_pct)}" y1="${y - 12}" y2="${y + 12}" style="stroke:var(--s3);stroke-width:2.5;stroke-linecap:round"/>`;
  s += `<circle cx="${X(fl.residual_pct)}" cy="${y}" r="6.5" style="fill:var(--s1);stroke:var(--card);stroke-width:2"/>`;
  const lx = X(fl.low_pct);
  const anchor = lx < 90 ? "start" : "middle";
  s += `<text x="${lx}" y="12" text-anchor="${anchor}" style="fill:var(--ink);font-size:11.5px;font-weight:600">lower bound ${fmt(fl.low_pct)} % (operative)</text>`;
  s += "</svg>";
  const wrap = el("div");
  wrap.innerHTML = s;
  const items = [["dot", "var(--s1)", "ANN estimate"], ["band", "color-mix(in srgb, var(--s1) 22%, transparent)", "90 % range"],
                 ["line", "var(--s3)", "IS 456 area loss", "height:10px;width:3px"]];
  if (fl.upper_truncated) items.push(["band", "transparent", "range above 100 % (not credited)", "border:1px dashed var(--s1)"]);
  wrap.append(legend(items));
  return wrap;
}

function curveChart(fl, corr) {
  const c = fl.curve;
  const W = 640, H = 270, m = { l: 46, r: 16, t: 12, b: 40 };
  const xmax = c.corr_pct[c.corr_pct.length - 1];
  const lows = [...c.ann_pct, ...c.is456_pct, fl.low_pct];
  const ymin = Math.min(50, Math.floor((Math.min(...lows) - 5) / 10) * 10);
  const ymax = 110;
  const X = (v) => m.l + (v / xmax) * (W - m.l - m.r);
  const Y = (v) => m.t + ((ymax - v) / (ymax - ymin)) * (H - m.t - m.b);
  const path = (ys) => ys.map((v, i) => `${i ? "L" : "M"}${X(c.corr_pct[i]).toFixed(1)},${Y(v).toFixed(1)}`).join("");

  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Residual flexural strength against corrosion for this section">`;
  for (let v = ymin; v <= ymax; v += 10) {
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" style="stroke:var(--grid)"/>`;
    s += `<text x="${m.l - 8}" y="${Y(v) + 4}" text-anchor="end" style="fill:var(--muted);font-size:11px">${v}</text>`;
  }
  const step = xmax > 40 ? 10 : 5;
  for (let v = 0; v <= xmax; v += step) {
    s += `<text x="${X(v)}" y="${H - m.b + 16}" text-anchor="middle" style="fill:var(--muted);font-size:11px">${v}</text>`;
  }
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" style="stroke:var(--axis)"/>`;
  s += `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle" style="fill:var(--ink-2);font-size:12px">Average corrosion of tension bars (% mass loss)</text>`;
  s += `<text transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle" style="fill:var(--ink-2);font-size:12px">Residual strength (%)</text>`;
  s += `<path d="${path(c.is456_pct)}" style="fill:none;stroke:var(--s3);stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>`;
  s += `<path d="${path(c.ann_pct)}" style="fill:none;stroke:var(--s2);stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>`;
  const cx = X(Math.min(corr, xmax));
  s += `<line x1="${cx}" x2="${cx}" y1="${Y(fl.high_pct)}" y2="${Y(fl.low_pct)}" style="stroke:var(--s1);stroke-width:2;stroke-linecap:round"/>`;
  s += `<line x1="${cx - 5}" x2="${cx + 5}" y1="${Y(fl.high_pct)}" y2="${Y(fl.high_pct)}" style="stroke:var(--s1);stroke-width:2;stroke-linecap:round"/>`;
  s += `<line x1="${cx - 5}" x2="${cx + 5}" y1="${Y(fl.low_pct)}" y2="${Y(fl.low_pct)}" style="stroke:var(--s1);stroke-width:2;stroke-linecap:round"/>`;
  s += `<circle cx="${cx}" cy="${Y(fl.residual_pct)}" r="6" style="fill:var(--s1);stroke:var(--card);stroke-width:2"/>`;
  s += `<line class="cross" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" style="stroke:var(--ink-2);stroke-width:1;opacity:0"/>`;
  s += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" style="fill:transparent"/>`;
  s += "</svg>";

  const wrap = el("div");
  wrap.innerHTML = s;
  wrap.append(legend([["line", "var(--s2)", "ANN (M3)"], ["line", "var(--s3)", "IS 456 (steel-area loss only)"],
                      ["dot", "var(--s1)", "Assessed state with 90 % range"]]));

  // Hover: crosshair + tooltip showing both curves at the nearest corrosion level.
  const svg = $("svg", wrap);
  const cross = $(".cross", wrap);
  const tip = el("div", undefined, "tooltip");
  tip.hidden = true;
  wrap.append(tip);
  $(".hit", wrap).addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const v = ((px - m.l) / (W - m.l - m.r)) * xmax;
    let i = 0;
    c.corr_pct.forEach((cv, k) => { if (Math.abs(cv - v) < Math.abs(c.corr_pct[i] - v)) i = k; });
    const x = X(c.corr_pct[i]);
    cross.setAttribute("x1", x);
    cross.setAttribute("x2", x);
    cross.style.opacity = 0.5;
    tip.innerHTML = "";
    tip.append(el("b", `Corrosion ${fmt(c.corr_pct[i], 0)} %`));
    [["var(--s2)", "ANN", c.ann_pct[i]], ["var(--s3)", "IS 456", c.is456_pct[i]]].forEach(([col, name, val]) => {
      const row = el("div", undefined, "row");
      const k = el("i", undefined, "key-line");
      k.style.background = col;
      row.append(k, document.createTextNode(`${name} ${fmt(val)} %`));
      tip.append(row);
    });
    tip.hidden = false;
    tip.style.left = `${(x / W) * r.width}px`;
    tip.style.top = `${(Y(Math.max(c.ann_pct[i], c.is456_pct[i])) / H) * r.height - 6}px`;
  });
  $(".hit", wrap).addEventListener("mouseleave", () => {
    cross.style.opacity = 0;
    tip.hidden = true;
  });
  return wrap;
}

init();
