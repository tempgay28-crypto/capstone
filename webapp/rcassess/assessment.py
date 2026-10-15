"""Assessment engine behind the web app: one beam in, one report out.

1. Corrosion state (condition.py): prior from exposure and age, updated with NDT and visual
   evidence. If a measured corrosion level is entered it is used directly; otherwise the
   corrosion level is inferred from the state probabilities ("full chain").
2. Residual flexural capacity from the ANN (M3), averaged over the corrosion distribution. Its
   90 % range combines three independent sources of uncertainty, each estimated from the data:
     diagnostic  spread of the corrosion level (measurement error or inferred-state spread)
     model       error between studies that more / better data would reduce
                 (held-out test error minus specimen scatter, plus disagreement between networks)
     residual    scatter between nominally identical specimens in the same study (irreducible)
3. Shear and conformity checks (is456.py) and the rehabilitation strategy (rehab.py).

Runs on NumPy alone: the trained networks, their calibration and the training-data ranges are
read from model.json (written by ann_model/04_export_web_model.py).
"""
import json
from pathlib import Path

import numpy as np

from . import condition, is456, rehab
from .analytical import corroded_capacity, effective_depth, steel_area
from .ann import Ensemble

MODEL_JSON = Path(__file__).resolve().parent / "model.json"

Z90 = 1.645  # two-sided 90 % interval of a normal distribution
CORR_GRID = np.arange(0.0, 80.01, 0.5)  # % mass loss

# Typical 1-SD uncertainty of each way of establishing the corrosion level (percentage
# points of mass loss). Engineering assumptions - shown in the app and editable per case.
CORROSION_METHODS = {
    "gravimetric": ("Gravimetric mass loss of extracted bar (ASTM G1)", 1.0),
    "section": ("Measured section loss of exposed bars (caliper / scan)", 3.0),
    "faraday": ("Estimated from impressed current (Faraday's law)", 5.0),
    "judgement": ("Engineering judgement / visual estimate", 8.0),
}

# name: (label, unit, required, min, max)
FIELDS = {
    "b_mm": ("Width b", "mm", True, 50, 2000),
    "D_mm": ("Overall depth D", "mm", True, 75, 3000),
    "L_mm": ("Effective span L", "mm", True, 300, 30000),
    "cover_mm": ("Clear cover", "mm", True, 5, 150),
    "bar_dia_mm": ("Tension bar diameter", "mm", True, 6, 50),
    "n_bars": ("Number of tension bars", "", True, 1, 30),
    "fck_mpa": ("Concrete strength fck", "MPa", True, 10, 120),
    "fy_mpa": ("Steel yield strength fy", "MPa", True, 200, 700),
    "fiber_pct": ("Fibre content", "%", False, 0, 4),
    "stirrup_dia_mm": ("Stirrup diameter", "mm", False, 4, 20),
    "stirrup_legs": ("Stirrup legs", "", False, 1, 8),
    "stirrup_spacing_mm": ("Stirrup spacing", "mm", False, 25, 600),
    "fy_stirrup_mpa": ("Stirrup yield strength", "MPa", False, 200, 700),
    "age_years": ("Age", "years", False, 0, 200),
    "wc_ratio": ("Water-cement ratio", "", False, 0.2, 0.9),
    "corr_pct": ("Measured average corrosion of tension bars", "% mass loss", False, 0, 80),
    "corr_sd_pp": ("Uncertainty of corrosion level (1 SD)", "% points", False, 0, 30),
    "stirrup_corr_pct": ("Average corrosion of stirrups", "% mass loss", False, 0, 90),
    "half_cell_mv": ("Half-cell potential (CSE)", "mV", False, -1000, 300),
    "icorr_ua_cm2": ("Corrosion current density", "µA/cm²", False, 0, 100),
    "resistivity_kohm_cm": ("Surface resistivity", "kΩ·cm", False, 0, 10000),
    "upv_km_s": ("Ultrasonic pulse velocity", "km/s", False, 0.5, 7),
    "chloride_pct_cement": ("Chloride at bar depth", "% by mass of cement", False, 0, 10),
    "carbonation_mm": ("Carbonation depth", "mm", False, 0, 300),
    "crack_width_mm": ("Maximum surface crack width", "mm", False, 0, 20),
    "demand_moment_knm": ("Factored moment demand Mu", "kN·m", False, 0, 1e5),
    "demand_shear_kn": ("Factored shear demand Vu", "kN", False, 0, 1e5),
}
CHOICE_FIELDS = {ev["field"]: ev["bands"] for ev in condition.EVIDENCE.values() if "edges" not in ev}
TEXT_FIELDS = ["structure", "member_ref", "inspector", "inspection_date", "location"]

# Optional inputs that improve the assessment (counted for "input completeness").
COMPLETENESS_FIELDS = ["exposure_class", "age_years", "wc_ratio", "corr_pct", "stirrup_dia_mm", "stirrup_spacing_mm",
                       "crack_width_mm", "demand_moment_knm", "demand_shear_kn", "half_cell_mv", "icorr_ua_cm2",
                       "resistivity_kohm_cm", "upv_km_s", "chloride_pct_cement", "carbonation_mm",
                       "rust_staining", "crack_pattern", "cover_condition", "exposed_rebar", "water_ingress"]

# Model inputs checked against the range of the training data.
RANGE_CHECK = {"b_mm": "Width b", "D_mm": "Overall depth D", "L_mm": "Span L", "cover_mm": "Clear cover",
               "bar_dia_mm": "Bar diameter", "n_bars": "Number of bars", "fck_mpa": "fck",
               "corr_pct": "Corrosion level", "rho_pct": "Steel ratio ρ"}

EXAMPLES = {
    "measured": {
        "structure": "Example: laboratory-scale beam", "member_ref": "B-15",
        "b_mm": 200, "D_mm": 300, "L_mm": 3000, "cover_mm": 30, "bar_dia_mm": 16, "n_bars": 2,
        "fck_mpa": 30, "fy_mpa": 500, "fiber_pct": 0, "stirrup_dia_mm": 8, "stirrup_legs": 2,
        "stirrup_spacing_mm": 150, "fy_stirrup_mpa": 500, "exposure_class": "Severe", "age_years": 18,
        "wc_ratio": 0.5, "corr_pct": 15, "corr_method": "section", "corr_sd_pp": 3, "stirrup_corr_pct": 20,
        "half_cell_mv": -380, "rust_staining": "Moderate", "crack_pattern": "Longitudinal along bars",
        "cover_condition": "Cracked", "crack_width_mm": 0.4, "demand_moment_knm": 30, "demand_shear_kn": 50,
    },
    "inspection": {
        "structure": "Example: coastal jetty approach", "member_ref": "Beam B-4", "location": "Soffit, mid-span, seaward",
        "b_mm": 230, "D_mm": 450, "L_mm": 3500, "cover_mm": 25, "bar_dia_mm": 16, "n_bars": 3,
        "fck_mpa": 25, "fy_mpa": 415, "stirrup_dia_mm": 8, "stirrup_legs": 2, "stirrup_spacing_mm": 175,
        "fy_stirrup_mpa": 415, "exposure_class": "Very severe", "age_years": 24, "wc_ratio": 0.5,
        "half_cell_mv": -410, "resistivity_kohm_cm": 12, "upv_km_s": 3.4, "chloride_pct_cement": 0.65,
        "carbonation_mm": 12, "rust_staining": "Moderate", "crack_pattern": "Longitudinal along bars",
        "crack_width_mm": 0.6, "cover_condition": "Delaminated (hollow)", "exposed_rebar": "Not visible",
        "water_ingress": "Occasional", "demand_moment_knm": 55, "demand_shear_kn": 80,
    },
}


class InputError(ValueError):
    pass


def parse_inputs(raw):
    x, errors = {}, []
    for name, (label, unit, required, lo, hi) in FIELDS.items():
        v = raw.get(name)
        if v is None or (isinstance(v, str) and not v.strip()):
            if required:
                errors.append(f"{label} is required.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number.")
            continue
        if not lo <= v <= hi:
            errors.append(f"{label} must be between {lo:g} and {hi:g} {unit}".strip() + ".")
            continue
        x[name] = v
    for name, options in {**CHOICE_FIELDS, "exposure_class": is456.EXPOSURE_CLASSES,
                          "corr_method": list(CORROSION_METHODS)}.items():
        v = str(raw.get(name) or "").strip()
        if v and v not in options:
            errors.append(f"Unknown value for {name.replace('_', ' ')}.")
        x[name] = v or None
    for name in TEXT_FIELDS:
        x[name] = str(raw.get(name) or "").strip()[:120] or None
    if errors:
        raise InputError(" ".join(errors))
    if x["cover_mm"] >= x["D_mm"] / 2:
        raise InputError("Clear cover must be less than half the overall depth.")
    x.setdefault("fiber_pct", 0.0)
    x.setdefault("stirrup_legs", 2.0)
    x.setdefault("fy_stirrup_mpa", x["fy_mpa"])
    return x


class Assessor:
    def __init__(self, model_path=MODEL_JSON):
        spec = json.loads(Path(model_path).read_text(encoding="utf-8"))
        self.ensemble = Ensemble(spec)
        self.features = self.ensemble.features
        self.ranges = {c: tuple(spec["ranges"][c]) for c in RANGE_CHECK}
        self.cal = spec["calibration"]
        self.provenance = spec["provenance"]

    def _features(self, x, corr_values):
        """Model inputs for this section at each corrosion level, columns in the trained order."""
        ast = steel_area(x["bar_dia_mm"], x["n_bars"])
        d = effective_depth(x["D_mm"], x["cover_mm"], x["bar_dia_mm"])
        row = {"b_mm": x["b_mm"], "D_mm": x["D_mm"], "L_mm": x["L_mm"], "cover_mm": x["cover_mm"],
               "bar_dia_mm": x["bar_dia_mm"], "n_bars": x["n_bars"], "fck_mpa": x["fck_mpa"],
               "fiber_pct": x["fiber_pct"], "ast_mm2": ast, "d_eff_mm": d, "rho_pct": 100 * ast / (x["b_mm"] * d),
               "L_over_D": x["L_mm"] / x["D_mm"]}
        corr_values = np.asarray(corr_values, dtype=float)
        X = np.empty((len(corr_values), len(self.features)))
        for j, name in enumerate(self.features):
            X[:, j] = corr_values if name == "corr_pct" else row[name]
        return X

    def _ann(self, x, corr_values):
        """Mean and spread of the ensemble members' residual-strength predictions."""
        members = self.ensemble.predict_members(self._features(x, corr_values))
        return members.mean(axis=0), members.std(axis=0)

    @staticmethod
    def corrosion_belief(x, cond):
        """Probability weights of the corrosion level on CORR_GRID."""
        if x.get("corr_pct") is not None:
            c = x["corr_pct"]
            sd = x.get("corr_sd_pp")
            if sd is None:
                sd = CORROSION_METHODS[x["corr_method"] or "judgement"][1]
            if sd < 0.25:
                grid, w = np.array([c]), np.array([1.0])
            else:
                grid = CORR_GRID
                w = np.exp(-0.5 * ((grid - c) / sd) ** 2)
                w /= w.sum()
            source = "measured"
        else:
            grid = condition.MASS_LOSS_GRID
            w = condition.mass_loss_grid(cond["posterior"], grid)
            source = "inferred"
        mean = float(w @ grid)
        sd = float(np.sqrt(w @ (grid - mean) ** 2))
        return {"grid": grid, "weights": w, "mean": mean, "sd": sd, "source": source}

    def flexure(self, x, belief):
        keep = belief["weights"] > 1e-6
        cs, w = belief["grid"][keep], belief["weights"][keep] / belief["weights"][keep].sum()
        r_grid, _ = self._ann(x, cs)
        r = float(w @ r_grid)
        diagnostic = float(np.sqrt(w @ (r_grid - r) ** 2))

        # Uncorroded beams scatter less than corroded ones; blend from 0 to 5 % corrosion.
        c = belief["mean"]
        _, spread = self._ann(x, [c])
        blend = float(np.clip(c / 5.0, 0, 1))
        cal = self.cal
        scale = (1 - blend) * cal["rmse_controls"] / cal["rmse_corroded"] + blend
        residual = scale * cal["replicate_sd"]
        model = float(np.sqrt((scale * cal["model_sd"]) ** 2 + spread[0] ** 2))
        total = float(np.sqrt(diagnostic ** 2 + model ** 2 + residual ** 2))
        lo_raw, hi_raw = r - Z90 * total, r + Z90 * total
        lo, hi = max(lo_raw, 0.0), min(hi_raw, 100.0)

        code = is456.flexure_is456(x)
        mu0 = code["mu0_knm"]
        if belief["source"] == "measured":
            diag_what = f"Corrosion level ±{belief['sd']:.1f} points; measure mass loss on extracted bars"
        else:
            diag_what = (f"Corrosion inferred from evidence (±{belief['sd']:.1f} points); "
                         "measure mass loss on an extracted bar")
        return {
            "residual_pct": r, "low_pct": lo, "high_pct": hi, "high_raw_pct": hi_raw,
            "upper_truncated": hi_raw > 100, "mu0_knm": mu0, "corrosion_source": belief["source"],
            "mu_res_knm": r / 100 * mu0, "mu_low_knm": lo / 100 * mu0, "mu_high_knm": hi / 100 * mu0,
            "area_loss_residual_pct": code["area_loss_residual_pct"], "mu_area_loss_knm": code["mu_area_loss_knm"],
            "uncertainty": [
                {"source": "Diagnostic", "sd": diagnostic, "what": diag_what},
                {"source": "Model", "sd": model, "what": "More test data of this kind (similar beams, same inputs)"},
                {"source": "Residual", "sd": residual, "what": "Not reducible: scatter between identical specimens"},
            ],
            "total_sd": total,
            "curve": self._curve(x),
        }

    def _curve(self, x):
        upper = max(40, 5 * int(np.ceil((x["corr_pct"] + 5) / 5)))
        cs = np.arange(0, upper + 1, 2.0)
        mean, _ = self._ann(x, cs)
        ana = corroded_capacity(x["b_mm"], x["D_mm"], x["cover_mm"], x["bar_dia_mm"], x["n_bars"],
                                x["fck_mpa"], cs, x["L_mm"], x["fy_mpa"])["residual_pct"]
        return {"corr_pct": cs.tolist(), "ann_pct": mean.tolist(), "is456_pct": np.asarray(ana).tolist()}

    def cautions(self, x, flex, shear, cond, belief):
        out = []
        n_evidence = cond["n_instruments"] + cond["n_observations"]
        if belief["source"] == "inferred":
            if n_evidence == 0:
                out.append("No corrosion measurement and no NDT or visual evidence entered: the corrosion level "
                           "reflects only the exposure-and-age prior, so the result is very uncertain.")
            else:
                out.append(f"Corrosion level inferred from {n_evidence} pieces of NDT and visual evidence "
                           f"({cond['state']}, probability {cond['state_prob']:.2f}; expected mass loss "
                           f"{belief['mean']:.1f} ± {belief['sd']:.1f} %). The evidence model is engineering judgement "
                           "anchored to published thresholds, not fitted to data - confirm with a measured mass loss.")
        else:
            measured_state = condition.state_of(x["corr_pct"])
            p_state = float(cond["evidence_only"][condition.STATES.index(measured_state)])
            if n_evidence >= 2 and p_state < 0.10:
                out.append(f"The NDT and visual evidence disagrees with the measured corrosion level: it gives only "
                           f"{p_state:.2f} probability to '{measured_state}' and points to "
                           f"'{condition.STATES[int(np.argmax(cond['evidence_only']))]}'. The sample may not represent "
                           "the member - check where it was taken. The rehabilitation rules act on the combined, worse "
                           "picture.")
            if x["corr_method"] in (None, "judgement"):
                out.append("The corrosion level is a judgement, not a measurement - it is the input the result is "
                           "most sensitive to. Measure mass loss on extracted bars to narrow the range.")
        ast = steel_area(x["bar_dia_mm"], x["n_bars"])
        d = effective_depth(x["D_mm"], x["cover_mm"], x["bar_dia_mm"])
        values = {**x, "rho_pct": 100 * ast / (x["b_mm"] * d)}
        outside = [f"{label} {values[k]:g} (data {lo:g}–{hi:g})" for k, label in RANGE_CHECK.items()
                   for lo, hi in [self.ranges[k]] if not lo <= values[k] <= hi]
        if outside:
            out.append("Outside the range of the beams the ANN was trained on: " + "; ".join(outside)
                       + ". The learned estimate is an extrapolation.")
        if x["corr_pct"] > 0.5 and self.cal["bias_corroded"] > 1:
            out.append(f"On held-out studies the ANN overestimated corroded beams by {self.cal['bias_corroded']:.1f} "
                       "points on average. Use the lower bound as the operative figure.")
        if flex["upper_truncated"]:
            out.append(f"The upper bound reached {flex['high_raw_pct']:.0f} % before truncation at 100 %: on this "
                       "evidence the member cannot be distinguished from an undamaged one. The lower bound governs.")
        out.append("Steel fy is not an input of the ANN (the training data do not record it); it is used only in the "
                   "IS 456 baseline that converts the residual % to kN·m.")
        if not (x.get("stirrup_dia_mm") and x.get("stirrup_spacing_mm")):
            out.append("No stirrups entered: shear capacity is the concrete contribution only.")
        out.append("Shear is code mechanics with section loss only. It ignores loss of bond and anchorage and "
                   "pitting at stirrup bends, so it is not a validated model for corroded members.")
        if shear["after"]["capped"]:
            out.append("Shear capacity is limited by the maximum shear stress τc,max (Table 20), not by the stirrups.")
        return out

    def assess(self, raw):
        x = parse_inputs(raw)
        cond = condition.assess(x)
        belief = self.corrosion_belief(x, cond)
        x["corr_measured"] = belief["source"] == "measured"
        x["corr_pct"], x["corr_sd_pp"] = (x["corr_pct"], belief["sd"]) if x["corr_measured"] else (belief["mean"], belief["sd"])
        x.setdefault("stirrup_corr_pct", x["corr_pct"])

        flex = self.flexure(x, belief)
        shear = is456.corroded_shear(x)
        checks = is456.conformity(x, flex, shear)
        # Count what the user entered, not the defaults filled in above.
        given = sum(str(raw.get(k) or "").strip() != "" for k in COMPLETENESS_FIELDS)
        return clean({
            "inputs": x,
            "condition": cond,
            "corrosion": {k: belief[k] for k in ("mean", "sd", "source")},
            "flexure": flex,
            "shear": shear,
            "checks": checks,
            "rehabilitation": rehab.recommend(x, flex, shear, checks, cond),
            "cautions": self.cautions(x, flex, shear, cond, belief),
            "completeness": {"given": given, "of": len(COMPLETENESS_FIELDS),
                             "pct": 100 * given / len(COMPLETENESS_FIELDS)},
            "provenance": self.provenance,
        })

    def meta(self):
        return clean({"fields": {k: {"label": v[0], "unit": v[1], "required": v[2], "min": v[3], "max": v[4]}
                                 for k, v in FIELDS.items()},
                      "exposure_classes": is456.EXPOSURE_CLASSES,
                      "choices": CHOICE_FIELDS,
                      "corrosion_methods": {k: {"label": v[0], "sd": v[1]} for k, v in CORROSION_METHODS.items()},
                      "examples": EXAMPLES, "provenance": self.provenance})


def clean(obj):
    """Make numpy values JSON-serialisable and round floats for display."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), 3)
    return obj
