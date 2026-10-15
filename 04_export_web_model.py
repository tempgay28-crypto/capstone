"""Step 4 - Export the residual-strength ANN (M3) for the web app.

Writes webapp/rcassess/model.json with
  * the weights and preprocessing of every network in the M3 ensemble,
  * the uncertainty calibration (held-out test error, specimen scatter, bias),
  * the input ranges of the training data and the model's provenance,
then checks that the NumPy forward pass (webapp/rcassess/ann.py) reproduces scikit-learn.
Re-run after retraining M3 (02_train_models.py), then redeploy the web app.

Run:  python 04_export_web_model.py
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
from webapp.rcassess.ann import Ensemble

warnings.filterwarnings("ignore")

MODEL = "M3_beam_residual_strength"
OUT = config.ROOT / "webapp" / "rcassess" / "model.json"
RANGE_COLS = ["b_mm", "D_mm", "L_mm", "cover_mm", "bar_dia_mm", "n_bars", "fck_mpa", "corr_pct", "rho_pct"]


def export_network(est):
    """One fitted TransformedTargetRegressor(Pipeline(prep, mlp)) -> plain numbers."""
    prep = est.regressor_.named_steps["prep"]
    mlp = est.regressor_.named_steps["mlp"]
    blocks = []
    for name, pipe, cols in prep.transformers_:
        if name == "remainder":
            assert pipe == "drop" or len(cols) == 0, "unexpected passthrough columns"
            continue
        steps = pipe.named_steps
        if "log" in steps:
            assert steps["log"].func is np.log1p
        imp, sc = steps["impute"], steps["scale"]
        blocks.append({"cols": list(cols), "log": "log" in steps, "medians": imp.statistics_.tolist(),
                       "indicator": imp.indicator_.features_.tolist() if imp.indicator_ is not None else [],
                       "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()})
    assert mlp.out_activation_ == "identity"
    tr = est.transformer_
    if isinstance(tr, StandardScaler):
        y_scaler, log_target = tr, False
    else:
        assert tr.named_steps["log"].func is np.log1p
        y_scaler, log_target = tr.named_steps["scale"], True
    return {"blocks": blocks, "activation": mlp.activation,
            "weights": [w.tolist() for w in mlp.coefs_], "biases": [b.tolist() for b in mlp.intercepts_],
            "y_mean": float(y_scaler.mean_[0]), "y_scale": float(y_scaler.scale_[0]), "log_target": log_target}


def calibrate(beams, test):
    """Error sizes used for the 90 % range (see webapp/rcassess/assessment.py)."""
    err = test["ann_pred"] - test["residual_pct"]
    ctrl = test["corr_pct"] <= 0.5
    rmse_ctrl = float(np.sqrt((err[ctrl] ** 2).mean()))
    rmse_corr = float(np.sqrt((err[~ctrl] ** 2).mean()))
    b = beams.dropna(subset=["residual_pct", "corr_pct"])
    b = b[b["corr_pct"] > 0.5]
    devs = []  # deviations within groups of nominally identical specimens (same paper and section, corrosion within 2 %)
    for _, g in b.groupby(["source_id", "b_mm", "D_mm", "L_mm", "bar_dia_mm", "n_bars", "fck_mpa"], dropna=False):
        g = g.sort_values("corr_pct")
        for _, r in g.groupby((g["corr_pct"].diff() > 2).cumsum()):
            if len(r) >= 2:
                devs.extend((r["residual_pct"] - r["residual_pct"].mean()) * np.sqrt(len(r) / (len(r) - 1)))
    replicate = min(float(np.sqrt(np.mean(np.square(devs)))), rmse_corr)
    return {"rmse_controls": rmse_ctrl, "rmse_corroded": rmse_corr, "bias_corroded": float(err[~ctrl].mean()),
            "replicate_sd": replicate, "model_sd": float(np.sqrt(rmse_corr ** 2 - replicate ** 2)),
            "replicate_specimens": len(devs)}


def main():
    bundle = joblib.load(config.MODEL_DIR / f"{MODEL}.joblib")
    beams = pd.read_csv(config.BEAM_CLEAN_CSV)
    test = pd.read_csv(config.OUT_DIR / "test_predictions" / f"{MODEL}.csv")
    metrics = json.loads(config.METRICS_JSON.read_text())["models"][MODEL]

    cal = calibrate(beams, test)
    spec = {
        "model": MODEL,
        "features": bundle["features"],
        "members": [export_network(est) for est in bundle["model"].estimators_],
        "calibration": cal,
        "ranges": {c: [float(v) for v in beams[c].quantile([0.01, 0.99])] for c in RANGE_COLS},
        "provenance": {
            "rows_train": metrics["rows_train"], "rows_test": metrics["rows_test"],
            "papers_train": metrics["papers_train"], "papers_test": metrics["papers_test"],
            "test_r2": metrics["test"]["ANN"]["r2"], "test_rmse": metrics["test"]["ANN"]["rmse"],
            "trained_rows": bundle["trained_rows"], **{k: round(v, 2) for k, v in cal.items()},
        },
    }

    # The NumPy forward pass must reproduce scikit-learn (test rows include missing inputs).
    X = test[bundle["features"]]
    ens = Ensemble(spec)
    sk_members = np.array([m.predict(X) for m in bundle["model"].estimators_])
    np_members = ens.predict_members(X.to_numpy(dtype=float))
    diff = float(np.abs(sk_members - np_members).max())
    assert diff < 1e-6, f"NumPy and scikit-learn predictions differ by {diff}"

    OUT.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB): {len(spec['members'])} networks, "
          f"{len(spec['features'])} inputs; NumPy vs scikit-learn max difference {diff:.2e} on {len(X)} test rows.")


if __name__ == "__main__":
    main()
