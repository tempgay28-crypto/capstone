"""Assessment engine behind the web app: one beam in, one report out.

Residual flexural capacity comes from the ANN (M3); its 90 % range combines three
independent sources of uncertainty, each estimated from the data:

  diagnostic  how uncertain the entered corrosion level is, pushed through the ANN
  model       error between studies that more / better data would reduce
              (held-out test error minus specimen scatter, plus disagreement between networks)
  residual    scatter between nominally identical specimens in the same study (irreducible)
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import config
import is456
from analytical import effective_depth, steel_area

warnings.filterwarnings("ignore")

Z90 = 1.645  # two-sided 90 % interval of a normal distribution

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
    "corr_pct": ("Average corrosion of tension bars", "% mass loss", True, 0, 80),
    "corr_sd_pp": ("Uncertainty of corrosion level (1 SD)", "% points", False, 0, 30),
    "stirrup_corr_pct": ("Average corrosion of stirrups", "% mass loss", False, 0, 90),
    "crack_width_mm": ("Maximum surface crack width", "mm", False, 0, 20),
    "demand_moment_knm": ("Factored moment demand Mu", "kN·m", False, 0, 1e5),
    "demand_shear_kn": ("Factored shear demand Vu", "kN", False, 0, 1e5),
}
# Optional inputs that improve the assessment (counted for "input completeness").
COMPLETENESS_FIELDS = ["exposure_class", "age_years", "wc_ratio", "corr_method", "stirrup_dia_mm",
                       "stirrup_spacing_mm", "stirrup_corr_pct", "crack_width_mm",
                       "demand_moment_knm", "demand_shear_kn"]

# Model inputs checked against the range of the training data.
RANGE_CHECK = {"b_mm": "Width b", "D_mm": "Overall depth D", "L_mm": "Span L", "cover_mm": "Clear cover",
               "bar_dia_mm": "Bar diameter", "n_bars": "Number of bars", "fck_mpa": "fck",
               "corr_pct": "Corrosion level", "rho_pct": "Steel ratio ρ"}

WORKED_EXAMPLE = {
    "b_mm": 200, "D_mm": 300, "L_mm": 3000, "cover_mm": 30, "bar_dia_mm": 16, "n_bars": 2,
    "fck_mpa": 30, "fy_mpa": 500, "fiber_pct": 0, "stirrup_dia_mm": 8, "stirrup_legs": 2,
    "stirrup_spacing_mm": 150, "fy_stirrup_mpa": 500, "exposure_class": "Severe", "age_years": 18,
    "wc_ratio": 0.5, "corr_pct": 15, "corr_method": "section", "corr_sd_pp": 3, "stirrup_corr_pct": 20,
    "crack_width_mm": 0.4, "demand_moment_knm": 30, "demand_shear_kn": 50,
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
    exp = (raw.get("exposure_class") or "").strip()
    if exp and exp not in is456.EXPOSURE_CLASSES:
        errors.append("Unknown exposure class.")
    x["exposure_class"] = exp or None
    method = (raw.get("corr_method") or "").strip()
    if method and method not in CORROSION_METHODS:
        errors.append("Unknown corrosion measurement method.")
    x["corr_method"] = method or None
    if errors:
        raise InputError(" ".join(errors))

    if x["cover_mm"] >= x["D_mm"] / 2:
        raise InputError("Clear cover must be less than half the overall depth.")
    x.setdefault("fiber_pct", 0.0)
    x.setdefault("stirrup_legs", 2.0)
    x.setdefault("fy_stirrup_mpa", x["fy_mpa"])
    x.setdefault("stirrup_corr_pct", x["corr_pct"])
    if "corr_sd_pp" not in x:
        x["corr_sd_pp"] = CORROSION_METHODS[x["corr_method"] or "judgement"][1]
    return x


class Assessor:
    def __init__(self):
        bundle = joblib.load(config.MODEL_DIR / "M3_beam_residual_strength.joblib")
        self.model, self.features = bundle["model"], bundle["features"]
        beams = pd.read_csv(config.BEAM_CLEAN_CSV)
        self.ranges = {c: tuple(beams[c].quantile([0.01, 0.99])) for c in RANGE_CHECK}
        metrics = json.loads(config.METRICS_JSON.read_text())["models"]["M3_beam_residual_strength"]
        self.cal = self._calibrate(beams)
        self.provenance = {
            "rows_train": metrics["rows_train"], "rows_test": metrics["rows_test"],
            "papers_train": metrics["papers_train"], "papers_test": metrics["papers_test"],
            "test_r2": metrics["test"]["ANN"]["r2"], "test_rmse": metrics["test"]["ANN"]["rmse"],
            "trained_rows": bundle["trained_rows"], **{k: round(v, 2) for k, v in self.cal.items()},
        }

    @staticmethod
    def _calibrate(beams):
        """Error sizes from held-out test papers and replicate specimens (see module docstring)."""
        t = pd.read_csv(config.OUT_DIR / "test_predictions" / "M3_beam_residual_strength.csv")
        err = t["ann_pred"] - t["residual_pct"]
        ctrl = t["corr_pct"] <= 0.5
        rmse_ctrl = float(np.sqrt((err[ctrl] ** 2).mean()))
        rmse_corr = float(np.sqrt((err[~ctrl] ** 2).mean()))

        b = beams.dropna(subset=["residual_pct", "corr_pct"])
        b = b[b["corr_pct"] > 0.5]
        devs = []
        for _, g in b.groupby(["source_id", "b_mm", "D_mm", "L_mm", "bar_dia_mm", "n_bars", "fck_mpa"], dropna=False):
            g = g.sort_values("corr_pct")
            for _, r in g.groupby((g["corr_pct"].diff() > 2).cumsum()):  # corrosion within 2 points
                if len(r) >= 2:
                    devs.extend((r["residual_pct"] - r["residual_pct"].mean()) * np.sqrt(len(r) / (len(r) - 1)))
        replicate = float(np.sqrt(np.mean(np.square(devs))))
        replicate = min(replicate, rmse_corr)
        return {"rmse_controls": rmse_ctrl, "rmse_corroded": rmse_corr, "bias_corroded": float(err[~ctrl].mean()),
                "replicate_sd": replicate, "model_sd": float(np.sqrt(rmse_corr ** 2 - replicate ** 2)),
                "replicate_specimens": len(devs)}

    def _features(self, x, corr_values):
        ast = steel_area(x["bar_dia_mm"], x["n_bars"])
        d = effective_depth(x["D_mm"], x["cover_mm"], x["bar_dia_mm"])
        row = {"b_mm": x["b_mm"], "D_mm": x["D_mm"], "L_mm": x["L_mm"], "cover_mm": x["cover_mm"],
               "bar_dia_mm": x["bar_dia_mm"], "n_bars": x["n_bars"], "fck_mpa": x["fck_mpa"],
               "fiber_pct": x["fiber_pct"], "ast_mm2": ast, "d_eff_mm": d, "rho_pct": 100 * ast / (x["b_mm"] * d),
               "L_over_D": x["L_mm"] / x["D_mm"]}
        X = pd.DataFrame([row] * len(corr_values))
        X["corr_pct"] = corr_values
        return X[self.features]

    def _ann(self, x, corr_values):
        """Mean and spread of the ensemble members' residual-strength predictions."""
        X = self._features(x, np.asarray(corr_values, dtype=float))
        members = np.array([m.predict(X) for m in self.model.estimators_])
        return members.mean(axis=0), members.std(axis=0)

    def flexure(self, x):
        c, sd_c = x["corr_pct"], x["corr_sd_pp"]
        c_lo, c_hi = max(c - sd_c, 0.0), c + sd_c
        mean, spread = self._ann(x, [c, c_lo, c_hi])
        r = float(mean[0])
        slope = (mean[2] - mean[1]) / (c_hi - c_lo) if c_hi > c_lo else 0.0
        diagnostic = abs(slope) * sd_c

        # Uncorroded beams scatter less than corroded ones; blend from 0 to 5 % corrosion.
        w = float(np.clip(c / 5.0, 0, 1))
        cal = self.cal
        scale = (1 - w) * cal["rmse_controls"] / cal["rmse_corroded"] + w
        residual = scale * cal["replicate_sd"]
        model = float(np.sqrt((scale * cal["model_sd"]) ** 2 + spread[0] ** 2))
        total = float(np.sqrt(diagnostic ** 2 + model ** 2 + residual ** 2))
        lo_raw, hi_raw = r - Z90 * total, r + Z90 * total
        lo, hi = max(lo_raw, 0.0), min(hi_raw, 100.0)

        code = is456.flexure_is456(x)
        mu0 = code["mu0_knm"]
        return {
            "residual_pct": r, "low_pct": lo, "high_pct": hi, "high_raw_pct": hi_raw,
            "upper_truncated": hi_raw > 100, "mu0_knm": mu0,
            "mu_res_knm": r / 100 * mu0, "mu_low_knm": lo / 100 * mu0, "mu_high_knm": hi / 100 * mu0,
            "area_loss_residual_pct": code["area_loss_residual_pct"], "mu_area_loss_knm": code["mu_area_loss_knm"],
            "uncertainty": [
                {"source": "Diagnostic", "sd": diagnostic,
                 "what": f"Corrosion level ±{sd_c:g} points; measure mass loss on extracted bars"},
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
        from analytical import corroded_capacity
        ana = corroded_capacity(x["b_mm"], x["D_mm"], x["cover_mm"], x["bar_dia_mm"], x["n_bars"],
                                x["fck_mpa"], cs, x["L_mm"], x["fy_mpa"])["residual_pct"]
        return {"corr_pct": cs.tolist(), "ann_pct": mean.tolist(), "is456_pct": np.asarray(ana).tolist()}

    def cautions(self, x, flex, shear):
        out = []
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
        if x["corr_method"] in (None, "judgement"):
            out.append("The corrosion level is a judgement, not a measurement - it is the input the result is most "
                       "sensitive to. Measure mass loss on extracted bars to narrow the range.")
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
        flex = self.flexure(x)
        shear = is456.corroded_shear(x)
        # Count what the user entered, not the defaults parse_inputs filled in.
        given = sum(str(raw.get(k) or "").strip() != "" for k in COMPLETENESS_FIELDS)
        return clean({
            "inputs": x,
            "flexure": flex,
            "shear": shear,
            "checks": is456.conformity(x, flex, shear),
            "cautions": self.cautions(x, flex, shear),
            "completeness": {"given": given, "of": len(COMPLETENESS_FIELDS),
                             "pct": 100 * given / len(COMPLETENESS_FIELDS)},
            "provenance": self.provenance,
        })

    def meta(self):
        return clean({"fields": {k: {"label": v[0], "unit": v[1], "required": v[2], "min": v[3], "max": v[4]}
                                 for k, v in FIELDS.items()},
                      "exposure_classes": is456.EXPOSURE_CLASSES,
                      "corrosion_methods": {k: {"label": v[0], "sd": v[1]} for k, v in CORROSION_METHODS.items()},
                      "worked_example": WORKED_EXAMPLE, "provenance": self.provenance})


def clean(obj):
    """Make numpy values JSON-serialisable and round floats for display."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), 3)
    return obj
