"""Corrosion state of a member from exposure, NDT and visual inspection (Bayesian evidence model).

Five states, defined by average mass loss of the tension bars:
    Negligible < 1 %, Low 1-5 %, Moderate 5-10 %, High 10-20 %, Severe 20-40 %

The prior depends on the IS 456 exposure class and age. Each reading falls into one of the
bands used by its standard (e.g. ASTM C876 half-cell bands), and each band has a probability of
being seen in each state (the LIKELIHOOD tables below). Posterior = prior x likelihoods.

The likelihoods are ENGINEERING JUDGEMENT anchored to the published band limits - they were not
fitted to data (the beam dataset cannot support that: model M1 scored R² ≈ 0). Treat the result
as a structured, transparent estimate and validate it against your own measured cases.

Visual signs (staining, cracking, spalling, exposed bars) all come from the same rust and are
therefore correlated; their combined weight is tempered so they are not counted several times.
"""
import numpy as np

STATES = ["Negligible", "Low", "Moderate", "High", "Severe"]
STATE_BINS = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 40)]  # % mass loss
VISUAL_TEMPERING = 0.75  # exponent applied to the product of the correlated visual likelihoods

PRIOR_BY_EXPOSURE = {
    None: [0.30, 0.25, 0.20, 0.15, 0.10],
    "Mild": [0.55, 0.25, 0.12, 0.06, 0.02],
    "Moderate": [0.40, 0.28, 0.17, 0.10, 0.05],
    "Severe": [0.28, 0.27, 0.22, 0.15, 0.08],
    "Very severe": [0.20, 0.24, 0.24, 0.20, 0.12],
    "Extreme": [0.14, 0.20, 0.25, 0.24, 0.17],
}
AGE_FACTOR = [  # (upper age limit in years, multiplier per state)
    (10, [1.4, 1.2, 0.9, 0.6, 0.4]),
    (25, [1.0, 1.0, 1.0, 1.0, 1.0]),
    (np.inf, [0.7, 0.9, 1.05, 1.2, 1.3]),
]

# Each evidence item: input field, group, how readings map to bands, and P(band | state).
# Rows of "lik" follow STATES; columns follow "bands".
EVIDENCE = {
    "half_cell": dict(
        field="half_cell_mv", group="instrument", label="Half-cell potential (CSE)", unit="mV",
        test="Half-cell potential mapping", kind="NDT", cost="low",
        source="ASTM C876 bands: > −200 mV ~90 % no activity; < −350 mV ~90 % active corrosion",
        edges=[-500, -350, -200], bands=["< −500 mV", "−500 to −350 mV", "−350 to −200 mV", "> −200 mV"],
        lik=[[0.02, 0.06, 0.22, 0.70],
             [0.08, 0.22, 0.40, 0.30],
             [0.17, 0.38, 0.33, 0.12],
             [0.30, 0.42, 0.22, 0.06],
             [0.40, 0.40, 0.16, 0.04]]),
    "icorr": dict(
        field="icorr_ua_cm2", group="instrument", label="Corrosion current density", unit="µA/cm²",
        test="Corrosion rate (linear polarisation)", kind="NDT", cost="medium",
        source="RILEM TC 154-EMC rate bands: < 0.1 negligible, 0.1-0.5 low, 0.5-1 moderate, > 1 high",
        edges=[0.1, 0.5, 1.0], bands=["< 0.1", "0.1-0.5", "0.5-1.0", "> 1.0"],
        lik=[[0.70, 0.22, 0.06, 0.02],
             [0.25, 0.45, 0.20, 0.10],
             [0.10, 0.35, 0.35, 0.20],
             [0.05, 0.20, 0.35, 0.40],
             [0.03, 0.12, 0.30, 0.55]]),
    "resistivity": dict(
        field="resistivity_kohm_cm", group="instrument", label="Surface resistivity (Wenner)", unit="kΩ·cm",
        test="Surface resistivity survey", kind="NDT", cost="low",
        source="RILEM TC 154-EMC: > 100 negligible, 50-100 low, 10-50 moderate, < 10 high corrosion rate",
        edges=[10, 50, 100], bands=["< 10", "10-50", "50-100", "> 100"],
        lik=[[0.05, 0.20, 0.30, 0.45],
             [0.10, 0.35, 0.30, 0.25],
             [0.20, 0.45, 0.23, 0.12],
             [0.30, 0.48, 0.15, 0.07],
             [0.40, 0.45, 0.10, 0.05]]),
    "upv": dict(
        field="upv_km_s", group="instrument", label="Ultrasonic pulse velocity", unit="km/s",
        test="Ultrasonic pulse velocity", kind="NDT", cost="low",
        source="IS 13311 (Part 1) grading: > 4.5 excellent, 3.5-4.5 good, 3.0-3.5 medium, < 3.0 doubtful",
        edges=[3.0, 3.5, 4.5], bands=["< 3.0", "3.0-3.5", "3.5-4.5", "> 4.5"],
        lik=[[0.03, 0.12, 0.50, 0.35],
             [0.05, 0.15, 0.50, 0.30],
             [0.10, 0.20, 0.48, 0.22],
             [0.17, 0.26, 0.42, 0.15],
             [0.25, 0.30, 0.35, 0.10]]),
    "chloride": dict(
        field="chloride_pct_cement", group="instrument", label="Chloride at bar depth", unit="% by mass of cement",
        test="Chloride profile (drilled dust samples)", kind="Semi-destructive", cost="medium",
        source="Concrete Society TR 54 risk bands: < 0.2 negligible, 0.2-0.4 low, 0.4-1.0 moderate, > 1.0 high",
        edges=[0.2, 0.4, 1.0], bands=["< 0.2", "0.2-0.4", "0.4-1.0", "> 1.0"],
        lik=[[0.55, 0.28, 0.13, 0.04],
             [0.25, 0.32, 0.30, 0.13],
             [0.12, 0.23, 0.40, 0.25],
             [0.07, 0.15, 0.40, 0.38],
             [0.05, 0.10, 0.35, 0.50]]),
    "carbonation": dict(
        field="carbonation_ratio", group="instrument", label="Carbonation depth / cover", unit="",
        test="Carbonation depth (phenolphthalein)", kind="Semi-destructive", cost="low",
        source="Depassivation when the carbonation front reaches the bar (ratio ≥ 1)",
        edges=[0.5, 1.0], bands=["< 0.5", "0.5-1.0", "≥ 1.0"],
        lik=[[0.65, 0.25, 0.10],
             [0.40, 0.32, 0.28],
             [0.28, 0.30, 0.42],
             [0.22, 0.28, 0.50],
             [0.18, 0.27, 0.55]]),
    "rust": dict(
        field="rust_staining", group="visual", label="Rust staining", unit="",
        test="Visual: rust staining", kind="Inspection", cost="very low",
        source="ACI 222R / ACI 201.1R condition survey signs",
        bands=["None", "Light", "Moderate", "Heavy"],
        lik=[[0.85, 0.12, 0.025, 0.005],
             [0.50, 0.35, 0.12, 0.03],
             [0.20, 0.35, 0.33, 0.12],
             [0.08, 0.20, 0.40, 0.32],
             [0.04, 0.10, 0.31, 0.55]]),
    "cracking": dict(
        field="crack_pattern", group="visual", label="Cracking pattern", unit="",
        test="Visual: crack mapping", kind="Inspection", cost="very low",
        source="Cracks along the bars are the typical sign of expanding rust",
        bands=["None", "Map / shrinkage", "Transverse (flexural)", "Longitudinal along bars"],
        lik=[[0.45, 0.25, 0.27, 0.03],
             [0.30, 0.22, 0.30, 0.18],
             [0.15, 0.15, 0.25, 0.45],
             [0.08, 0.10, 0.20, 0.62],
             [0.05, 0.07, 0.16, 0.72]]),
    "crack_width": dict(
        field="crack_width_mm", group="visual", label="Width of longitudinal cracks", unit="mm",
        test="Crack width (along bars)", kind="Inspection", cost="very low",
        source="Corrosion-crack width grows with section loss; used only for cracks along the bars",
        edges=[0.1, 0.3, 1.0], bands=["< 0.1", "0.1-0.3", "0.3-1.0", "> 1.0"],
        lik=[[0.80, 0.15, 0.04, 0.01],
             [0.45, 0.35, 0.16, 0.04],
             [0.20, 0.35, 0.35, 0.10],
             [0.08, 0.22, 0.45, 0.25],
             [0.04, 0.12, 0.39, 0.45]]),
    "cover": dict(
        field="cover_condition", group="visual", label="Cover condition (hammer tapping)", unit="",
        test="Hammer tapping / delamination survey", kind="Inspection", cost="very low",
        source="ACI 228.2R sounding; delamination and spalling follow rust expansion",
        bands=["Sound", "Cracked", "Delaminated (hollow)", "Spalled"],
        lik=[[0.85, 0.12, 0.02, 0.01],
             [0.55, 0.32, 0.09, 0.04],
             [0.25, 0.40, 0.22, 0.13],
             [0.10, 0.30, 0.30, 0.30],
             [0.04, 0.16, 0.30, 0.50]]),
    "exposed": dict(
        field="exposed_rebar", group="visual", label="Exposed reinforcement", unit="",
        test="Break-out inspection of a bar", kind="Inspection", cost="low",
        source="Condition of bars where visible",
        bands=["Not visible", "Surface rust only", "Pitting", "Visible section loss"],
        lik=[[0.96, 0.03, 0.008, 0.002],
             [0.80, 0.14, 0.05, 0.01],
             [0.60, 0.22, 0.13, 0.05],
             [0.40, 0.22, 0.20, 0.18],
             [0.25, 0.15, 0.22, 0.38]]),
    "water": dict(
        field="water_ingress", group="visual", label="Water ingress / dampness", unit="",
        test="Visual: leakage and dampness", kind="Inspection", cost="very low",
        source="Moisture sustains the corrosion cell",
        bands=["None", "Occasional", "Persistent"],
        lik=[[0.60, 0.30, 0.10],
             [0.45, 0.35, 0.20],
             [0.35, 0.38, 0.27],
             [0.28, 0.37, 0.35],
             [0.25, 0.35, 0.40]]),
}

# A measured mass loss is strong but not perfect evidence: one extracted bar may not represent the
# member, so it puts 0.9 on its own state and 0.025 on each other state.
EVIDENCE["measured"] = dict(
    field="corr_pct", group="measurement", label="Measured mass loss", unit="%",
    test="Mass loss of an extracted bar (ASTM G1)", kind="Destructive (local)", cost="high",
    source="Direct measurement; 0.9 weight on its own state because one bar may not represent the member",
    edges=[1, 5, 10, 20], bands=[f"{a}-{b} %" for a, b in STATE_BINS],
    lik=np.eye(5) * 0.875 + 0.025)

for _ev in EVIDENCE.values():
    _ev["lik"] = np.asarray(_ev["lik"], dtype=float)
    assert np.allclose(_ev["lik"].sum(axis=1), 1.0), _ev["label"]
    assert _ev["lik"].shape[1] == len(_ev["bands"]), _ev["label"]


def entropy_bits(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def kl_bits(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p > 0
    return float((p[m] * np.log2(p[m] / q[m])).sum())


def normalise(p):
    p = np.asarray(p, dtype=float)
    return p / p.sum()


def prior(exposure, age_years):
    p = np.array(PRIOR_BY_EXPOSURE.get(exposure, PRIOR_BY_EXPOSURE[None]), dtype=float)
    if age_years is not None:
        for limit, factor in AGE_FACTOR:
            if age_years < limit:
                p = p * np.array(factor)
                break
    return normalise(p)


def band_index(ev, value):
    """Index of the band a reading falls in (numeric edges or categorical labels)."""
    if "edges" in ev:
        return int(np.searchsorted(ev["edges"], value, side="right"))
    return ev["bands"].index(value)


def observations(x):
    """Evidence present in the inputs -> list of (key, band index, shown value)."""
    obs = []
    for key, ev in EVIDENCE.items():
        if key == "carbonation":
            if x.get("carbonation_mm") is None:
                continue
            value = x["carbonation_mm"] / x["cover_mm"]
            shown = f"{x['carbonation_mm']:g} mm = {value:.2f} × cover"
        elif key == "crack_width":
            # Only cracks along the bars say something about corrosion.
            if x.get("crack_width_mm") is None or x.get("crack_pattern") != "Longitudinal along bars":
                continue
            value = shown = x["crack_width_mm"]
            shown = f"{value:g} mm"
        else:
            value = x.get(ev["field"])
            if value is None:
                continue
            shown = f"{value:g} {ev['unit']}".strip() if isinstance(value, float) else value
        obs.append((key, band_index(ev, value), shown))
    return obs


def update(p, items):
    """Multiply the prior by the likelihood of each observation (visual group tempered)."""
    log_p = np.log(p)
    for key, band in items:
        ev = EVIDENCE[key]
        w = VISUAL_TEMPERING if ev["group"] == "visual" else 1.0
        log_p = log_p + w * np.log(ev["lik"][:, band])
    return normalise(np.exp(log_p - log_p.max()))


def expected_gain(p, lik):
    """Expected reduction in entropy (bits) from observing a test with likelihood table lik."""
    h0 = entropy_bits(p)
    p_out = p @ lik  # probability of each outcome
    gain = h0
    for j, po in enumerate(p_out):
        if po > 0:
            gain -= po * entropy_bits(normalise(p * lik[:, j]))
    return gain


def mass_loss_moments(p):
    """Mean and SD of mass loss for a state distribution (uniform within each state's range)."""
    mids = np.array([(a + b) / 2 for a, b in STATE_BINS])
    var_in = np.array([(b - a) ** 2 / 12 for a, b in STATE_BINS])
    mean = float(p @ mids)
    var = float(p @ (var_in + mids ** 2) - mean ** 2)
    return mean, float(np.sqrt(max(var, 0.0)))


# Cell centres every 0.5 % across 0-40 %: each state's range holds whole cells, so grid means match.
MASS_LOSS_GRID = np.arange(0.25, STATE_BINS[-1][1], 0.5)


def mass_loss_grid(p, grid=MASS_LOSS_GRID):
    """Probability weights of mass loss on a grid of cell centres (mixture of uniform state ranges)."""
    w = np.zeros_like(grid, dtype=float)
    for prob, (a, b) in zip(p, STATE_BINS):
        inside = (grid >= a) & (grid < b)
        if inside.any():
            w[inside] += prob / inside.sum()
    return w / w.sum()


def mass_loss_quantiles(p, qs):
    """Quantiles of the mixture, interpolated within the 0.5 % cells."""
    w = mass_loss_grid(p)
    cdf = np.concatenate([[0.0], np.cumsum(w)])
    edges = np.concatenate([MASS_LOSS_GRID - 0.25, [MASS_LOSS_GRID[-1] + 0.25]])
    return [float(np.interp(q, cdf, edges)) for q in qs]


def state_of(mass_loss):
    for name, (a, b) in zip(STATES, STATE_BINS):
        if mass_loss < b:
            return name
    return STATES[-1]


def assess(x):
    p0 = prior(x.get("exposure_class"), x.get("age_years"))
    obs = observations(x)
    instruments = [(k, b) for k, b, _ in obs if EVIDENCE[k]["group"] == "instrument"]
    visual = [(k, b) for k, b, _ in obs if EVIDENCE[k]["group"] == "visual"]
    measured = [(k, b) for k, b, _ in obs if EVIDENCE[k]["group"] == "measurement"]
    post = update(p0, instruments + visual + measured)
    post_evidence = update(p0, instruments + visual)  # NDT and visual only, to cross-check a measurement
    post_instr = update(p0, instruments)
    post_vis = update(p0, visual)
    best = int(np.argmax(post))
    mean, sd = mass_loss_moments(post)

    used = []
    for key, band, shown in obs:
        ev = EVIDENCE[key]
        used.append({"evidence": ev["label"], "value": shown, "band": ev["bands"][band],
                     "favours": STATES[int(np.argmax(ev["lik"][:, band]))], "group": ev["group"],
                     "source": ev["source"]})

    observed = {k for k, _, _ in obs}
    next_obs = []
    for key, ev in EVIDENCE.items():
        if key in observed or (key == "crack_width" and x.get("crack_pattern") != "Longitudinal along bars"):
            continue
        lik = ev["lik"] ** (VISUAL_TEMPERING if ev["group"] == "visual" else 1.0)
        lik = lik / lik.sum(axis=1, keepdims=True)
        next_obs.append({"test": ev["test"], "kind": ev["kind"], "cost": ev["cost"],
                         "gain_bits": expected_gain(post, lik)})
    next_obs.sort(key=lambda r: -r["gain_bits"])

    return {
        "states": STATES, "state_bins": STATE_BINS,
        "prior": p0, "posterior": post, "evidence_only": post_evidence,
        "instruments_only": post_instr, "observations_only": post_vis,
        "state": STATES[best], "state_prob": float(post[best]),
        "mass_loss_mean": mean, "mass_loss_sd": sd,
        "mass_loss_90": mass_loss_quantiles(post, [0.05, 0.95]),
        "entropy_bits": entropy_bits(post), "prior_entropy_bits": entropy_bits(p0),
        "info_instruments_bits": kl_bits(post_instr, p0), "n_instruments": len(instruments),
        "info_observations_bits": kl_bits(post_vis, p0), "n_observations": len(visual),
        "info_measurement_bits": kl_bits(update(p0, measured), p0), "n_measured": len(measured),
        "evidence_used": used, "next_observations": next_obs[:6],
    }
