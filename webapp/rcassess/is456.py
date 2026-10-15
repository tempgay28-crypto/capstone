"""IS 456:2000 shear capacity and conformity checks for an existing (corroded) beam.

Codal values are transcribed from IS 456:2000 (Tables 5, 16, 19, 20 and the clauses named
next to each check). Corrosion enters only as loss of bar area, so these are code-mechanics
estimates, not a validated model of corroded-member behaviour.
"""
import numpy as np

from .analytical import effective_depth, moment_capacity_knm, steel_area, xu_max_ratio

EXPOSURE_CLASSES = ["Mild", "Moderate", "Severe", "Very severe", "Extreme"]

# Table 16 (cl. 26.4.2.2): nominal cover to meet durability requirements, mm
NOMINAL_COVER_MM = {"Mild": 20, "Moderate": 30, "Severe": 45, "Very severe": 50, "Extreme": 75}
# Table 5 (cl. 6.1.2, 8.2.4.1): reinforced concrete, minimum grade and maximum free w/c ratio
MIN_GRADE_MPA = {"Mild": 20, "Moderate": 25, "Severe": 30, "Very severe": 35, "Extreme": 40}
MAX_WC_RATIO = {"Mild": 0.55, "Moderate": 0.50, "Severe": 0.45, "Very severe": 0.45, "Extreme": 0.40}

# Table 19: design shear strength of concrete tau_c (N/mm²) vs 100 As/(b d) and grade
_TC_PT = [0.15, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00]
_TC_GRADES = [15, 20, 25, 30, 35, 40]
_TC = np.array([
    [0.28, 0.28, 0.29, 0.29, 0.29, 0.30],
    [0.35, 0.36, 0.36, 0.37, 0.37, 0.38],
    [0.46, 0.48, 0.49, 0.50, 0.50, 0.51],
    [0.54, 0.56, 0.57, 0.59, 0.59, 0.60],
    [0.60, 0.62, 0.64, 0.66, 0.67, 0.68],
    [0.64, 0.67, 0.70, 0.71, 0.73, 0.74],
    [0.68, 0.72, 0.74, 0.76, 0.78, 0.79],
    [0.71, 0.75, 0.78, 0.80, 0.82, 0.84],
    [0.71, 0.79, 0.82, 0.84, 0.86, 0.88],
    [0.71, 0.81, 0.85, 0.88, 0.90, 0.92],
    [0.71, 0.82, 0.88, 0.91, 0.93, 0.95],
    [0.71, 0.82, 0.90, 0.94, 0.96, 0.98],
    [0.71, 0.82, 0.92, 0.96, 0.99, 1.01],
])
# Table 20: maximum shear stress tau_c,max (N/mm²)
_TC_MAX = [2.5, 2.8, 3.1, 3.5, 3.7, 4.0]


def tau_c(pt, fck):
    """Table 19, interpolated linearly in pt and grade (grades above M40 use M40)."""
    pt = float(np.clip(pt, _TC_PT[0], _TC_PT[-1]))
    fck = float(np.clip(fck, _TC_GRADES[0], _TC_GRADES[-1]))
    by_grade = [np.interp(pt, _TC_PT, _TC[:, j]) for j in range(len(_TC_GRADES))]
    return float(np.interp(fck, _TC_GRADES, by_grade))


def tau_c_max(fck):
    return float(np.interp(np.clip(fck, 15, 40), _TC_GRADES, _TC_MAX))


def required_cover_mm(exposure, fck, bar_dia_mm):
    """Table 16 with its notes 1 (mild, bars <= 12 mm: -5 mm) and 3 (severe/very severe, >= M35: -5 mm)."""
    cover = NOMINAL_COVER_MM[exposure]
    notes = []
    if exposure == "Mild" and bar_dia_mm <= 12:
        cover -= 5
        notes.append("reduced 5 mm for bars ≤ 12 mm in mild exposure (Table 16 note 1)")
    if exposure in ("Severe", "Very severe") and fck >= 35:
        cover -= 5
        notes.append("reduced 5 mm for M35 and above (Table 16 note 3)")
    return cover, notes


def crack_width_limit_mm(exposure, cover_mm):
    """Cl. 35.3.2: 0.3 mm in general, 0.2 mm where moisture makes cracking harmful, and
    0.004 x nominal cover in aggressive ('severe' and worse) environments."""
    if exposure is None or exposure == "Mild":
        return 0.3, "general limit"
    if exposure == "Moderate":
        return 0.2, "member exposed to moisture"
    return round(0.004 * cover_mm, 2), "0.004 × nominal cover, aggressive environment"


def shear_capacity(b, d, ast, fck, stirrup_dia, legs, spacing, fy_v):
    """Cl. 40: Vu = tau_c b d + 0.87 fy Asv d / sv, capped at tau_c,max b d. Returns kN and parts."""
    pt = 100 * ast / (b * d)
    vc = tau_c(pt, fck) * b * d / 1e3
    asv = legs * np.pi * stirrup_dia ** 2 / 4 if stirrup_dia and spacing else 0.0
    vs = 0.87 * fy_v * asv * d / spacing / 1e3 if asv else 0.0
    vmax = tau_c_max(fck) * b * d / 1e3
    return {"pt": pt, "tau_c": tau_c(pt, fck), "Vc_kn": vc, "Vs_kn": vs, "Vmax_kn": vmax,
            "Vu_kn": min(vc + vs, vmax), "capped": vc + vs > vmax, "asv_mm2": asv}


def corroded_shear(x):
    """Shear capacity before and after corrosion (tension bars and stirrups lose area)."""
    d = effective_depth(x["D_mm"], x["cover_mm"], x["bar_dia_mm"])
    ast0 = steel_area(x["bar_dia_mm"], x["n_bars"])
    ast_res = ast0 * (1 - x["corr_pct"] / 100)
    sdia = x.get("stirrup_dia_mm")
    sdia_res = sdia * np.sqrt(1 - x["stirrup_corr_pct"] / 100) if sdia else None  # equal-area diameter
    args = dict(b=x["b_mm"], d=d, fck=x["fck_mpa"], legs=x["stirrup_legs"],
                spacing=x.get("stirrup_spacing_mm"), fy_v=x["fy_stirrup_mpa"])
    before = shear_capacity(ast=ast0, stirrup_dia=sdia, **args)
    after = shear_capacity(ast=ast_res, stirrup_dia=sdia_res, **args)
    return {"before": before, "after": after, "d_mm": d,
            "retained_pct": 100 * after["Vu_kn"] / before["Vu_kn"]}


def _row(clause, check, required, actual, verdict, note=""):
    return {"clause": clause, "check": check, "required": required, "actual": actual,
            "verdict": verdict, "note": note}


def conformity(x, flex, shear):
    """IS 456 checks on the as-found member. Verdicts: compliant, deficient, exceeded,
    inadequate, marginal, informative, not checked."""
    rows = []
    exp, cover, fck, fy = x.get("exposure_class"), x["cover_mm"], x["fck_mpa"], x["fy_mpa"]
    b, D, bar = x["b_mm"], x["D_mm"], x["bar_dia_mm"]
    d = effective_depth(D, cover, bar)
    ast0 = steel_area(bar, x["n_bars"])
    ast_res = ast0 * (1 - x["corr_pct"] / 100)

    if exp:
        req, notes = required_cover_mm(exp, fck, bar)
        rows.append(_row("IS 456 cl. 26.4.2, Table 16", "Nominal cover for exposure class",
                         f"≥ {req} mm ({exp.lower()})", f"{cover:g} mm",
                         "compliant" if cover >= req else "deficient",
                         "; ".join(notes) or "Cover is the main barrier delaying chloride and carbonation reaching the steel."))
        g = MIN_GRADE_MPA[exp]
        rows.append(_row("IS 456 cl. 8.2.4.1, Table 5", "Minimum concrete grade for exposure class",
                         f"≥ M{g}", f"M{fck:g}", "compliant" if fck >= g else "deficient"))
        if x.get("wc_ratio") is not None:
            w = MAX_WC_RATIO[exp]
            rows.append(_row("IS 456 cl. 8.2.4.1, Table 5", "Maximum free water-cement ratio",
                             f"≤ {w:.2f}", f"{x['wc_ratio']:.2f}",
                             "compliant" if x["wc_ratio"] <= w else "exceeded"))
        else:
            rows.append(_row("IS 456 cl. 8.2.4.1, Table 5", "Maximum free water-cement ratio",
                             f"≤ {MAX_WC_RATIO[exp]:.2f}", "—", "not checked", "w/c ratio not entered."))
    else:
        rows.append(_row("IS 456 Tables 5 and 16", "Durability (cover, grade, w/c) for exposure class",
                         "—", "—", "not checked", "Exposure class not entered."))

    rows.append(_row("IS 456 cl. 26.4.1", "Cover not less than bar diameter", f"≥ {bar:g} mm", f"{cover:g} mm",
                     "compliant" if cover >= bar else "deficient"))

    if x.get("crack_width_mm") is not None:
        lim, why = crack_width_limit_mm(exp, cover)
        rows.append(_row("IS 456 cl. 35.3.2", "Surface crack width", f"≤ {lim:g} mm", f"{x['crack_width_mm']:g} mm",
                         "compliant" if x["crack_width_mm"] <= lim else "exceeded",
                         f"{why}. A service-load design limit, used here as a condition indicator."))
    else:
        rows.append(_row("IS 456 cl. 35.3.2", "Surface crack width", "—", "—", "not checked", "Crack width not entered."))

    as_min = 0.85 * b * d / fy
    rows.append(_row("IS 456 cl. 26.5.1.1(a)", "Minimum tension steel after section loss",
                     f"≥ {as_min:.0f} mm²", f"{ast_res:.0f} mm² ({100 - x['corr_pct']:.1f} % of {ast0:.0f})",
                     "compliant" if ast_res >= as_min else "deficient"))
    rows.append(_row("IS 456 cl. 26.5.1.1(b)", "Maximum tension steel", f"≤ {0.04 * b * D:.0f} mm²", f"{ast0:.0f} mm²",
                     "compliant" if ast0 <= 0.04 * b * D else "exceeded"))

    k = float(xu_max_ratio(fy))
    xu = 0.87 * fy * ast_res / (0.36 * fck * b)
    rows.append(_row("IS 456 cl. 38.1", "Under-reinforced section (ductile flexural failure)",
                     f"xu/d ≤ {k:.2f}", f"xu/d = {xu / d:.2f}",
                     "compliant" if xu / d <= k else "deficient",
                     "Corroded steel area used. An over-reinforced section fails by concrete crushing without warning."))

    sv, sdia = x.get("stirrup_spacing_mm"), x.get("stirrup_dia_mm")
    if sv and sdia:
        asv_res = shear["after"]["asv_mm2"]
        asv_min = 0.4 * b * sv / (0.87 * x["fy_stirrup_mpa"])
        rows.append(_row("IS 456 cl. 26.5.1.6", "Minimum shear reinforcement after section loss",
                         f"≥ {asv_min:.0f} mm²", f"{asv_res:.0f} mm² per set",
                         "compliant" if asv_res >= asv_min else "deficient"))
        smax = min(0.75 * d, 300)
        rows.append(_row("IS 456 cl. 26.5.1.5", "Maximum stirrup spacing", f"≤ {smax:.0f} mm", f"{sv:g} mm",
                         "compliant" if sv <= smax else "exceeded"))
    else:
        rows.append(_row("IS 456 cl. 26.5.1.5-26.5.1.6", "Shear reinforcement detailing", "—", "—", "not checked",
                         "Stirrup diameter and spacing not entered."))

    rows.append(_demand_row("IS 456 cl. 38.1, Annex G", "Flexural capacity vs factored moment",
                            x.get("demand_moment_knm"), flex["mu_low_knm"], flex["mu_res_knm"], "kN·m",
                            "Capacity = ANN residual % × IS 456 design capacity of the uncorroded section."))
    rows.append(_demand_row("IS 456 cl. 40.4", "Shear capacity vs factored shear",
                            x.get("demand_shear_kn"), shear["after"]["Vu_kn"], shear["after"]["Vu_kn"], "kN",
                            "Code mechanics with section loss only."))
    return rows


def _demand_row(clause, check, demand, low, central, unit, note):
    if demand is None:
        return _row(clause, check, "—", f"{central:.1f} {unit}", "informative", "No factored demand entered. " + note)
    if low >= demand:
        verdict = "compliant"
    elif central >= demand:
        verdict = "marginal"
        note = "Central estimate covers the demand but the 90 % lower bound does not. " + note
    else:
        verdict = "inadequate"
    actual = f"{central:.1f} {unit}" if low == central else f"{central:.1f} {unit} (lower bound {low:.1f})"
    return _row(clause, check, f"≥ {demand:g} {unit}", actual, verdict, note)


def flexure_is456(x):
    """IS 456 design moment capacity before corrosion and with steel-area loss only."""
    d = effective_depth(x["D_mm"], x["cover_mm"], x["bar_dia_mm"])
    ast0 = steel_area(x["bar_dia_mm"], x["n_bars"])
    mu0 = float(moment_capacity_knm(x["b_mm"], d, ast0, x["fck_mpa"], x["fy_mpa"]))
    mu_corr = float(moment_capacity_knm(x["b_mm"], d, ast0 * (1 - x["corr_pct"] / 100), x["fck_mpa"], x["fy_mpa"]))
    return {"d_mm": d, "ast0_mm2": ast0, "mu0_knm": mu0, "mu_area_loss_knm": mu_corr,
            "area_loss_residual_pct": 100 * mu_corr / mu0}
