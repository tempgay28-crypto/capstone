"""Step 2 - Train, tune and evaluate the ANN models.

Models
  M1   beam corrosion %                  <- exposure, chloride, half-cell potential, mix, cover
  M2   beam failure load (kN)            <- section, reinforcement, concrete grade, corrosion %
  M3   beam residual strength %          <- same inputs as M2
  M4a  rebar residual yield stress %     <- corrosion %, bar diameter, corrosion method
  M4b  rebar residual ultimate stress %  <- same inputs as M4a

For each model:
  1. Hold out ~20 % of the source papers as an unseen test set.
  2. Tune network size, activation and regularisation by 5-fold cross-validation in
     which every fold also keeps whole papers together (GroupKFold).
  3. Train an ensemble of networks (different random starts) with the best settings,
     score it on the held-out papers, next to a linear model, a random forest and
     (beams) the IS 456 calculation.
  4. Refit on all data and save to models/ for predict.py.

Run:  python 02_train_models.py                        (~30 min; uses all CPU cores)
      python 02_train_models.py M2_beam_failure_load   (retrain only the named model(s))
      python 02_train_models.py --figures              (redraw figures from saved results only)
"""
import json
import os
import sys
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # silence lbfgs convergence chatter in worker processes
# The networks are tiny, so one maths thread per process is fastest; parallelism comes
# from running many fits at once across CPU cores instead.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor, VotingRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler  # noqa: E402

import config  # noqa: E402
import plotting as P  # noqa: E402
from webapp.rcassess.analytical import corroded_capacity  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------- model definitions

BEAM_SECTION = ["b_mm", "D_mm", "L_mm", "cover_mm", "bar_dia_mm", "n_bars", "fck_mpa", "fiber_pct",
                "ast_mm2", "d_eff_mm", "rho_pct", "L_over_D"]
# Capacity scales multiplicatively with size (P ~ b d^2 / L), so sizes enter as logarithms.
BEAM_LOG = ["b_mm", "D_mm", "L_mm", "bar_dia_mm", "n_bars", "fck_mpa", "ast_mm2", "d_eff_mm", "rho_pct", "L_over_D"]

SPECS = {
    "M1_beam_corrosion": dict(
        dataset="beam", target="corr_pct", label="Average corrosion", unit="%", log_target=True,
        numeric=["exposure_days", "nacl_pct", "half_cell_mv", "cl_rebar_pct", "cover_mm", "wc_ratio",
                 "fck_mpa", "bar_dia_mm", "fiber_pct", "pre_crack_mm"],
        log_features=["exposure_days"], categorical=[], required=["exposure_days"]),
    "M2_beam_failure_load": dict(
        dataset="beam", target="load_kn", label="Failure load", unit="kN", log_target=True,
        numeric=BEAM_SECTION + ["corr_pct"], log_features=BEAM_LOG, categorical=[],
        required=["corr_pct", "b_mm", "D_mm"],
        only_consistent_loads=True),  # skip loads a flexural failure of the section cannot produce
    "M3_beam_residual_strength": dict(
        dataset="beam", target="residual_pct", label="Residual flexural strength", unit="%", log_target=False,
        numeric=BEAM_SECTION + ["corr_pct"], log_features=BEAM_LOG, categorical=[], required=["corr_pct"]),
    "M4a_rebar_residual_yield": dict(
        dataset="rebar", target="ry_eng_pct", label="Residual yield stress", unit="%", log_target=False,
        numeric=["corr_pct", "bar_dia_mm"], log_features=[], categorical=["method"], required=["corr_pct"]),
    "M4b_rebar_residual_ultimate": dict(
        dataset="rebar", target="ru_eng_pct", label="Residual ultimate stress", unit="%", log_target=False,
        numeric=["corr_pct", "bar_dia_mm"], log_features=[], categorical=["method"], required=["corr_pct"]),
}

PARAM_GRID = {
    # Small networks only: with ~800 noisy rows, bigger ones just memorise the training papers.
    "regressor__mlp__hidden_layer_sizes": [(4,), (8,), (16,), (16, 8), (32, 16)],
    "regressor__mlp__activation": ["relu", "tanh"],
    "regressor__mlp__alpha": [0.1, 1.0, 3.0, 10.0, 30.0],  # L2 penalty; data are noisy, so expect larger values
}


def preprocessor(spec):
    def numeric_steps(log):
        steps = [("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] if log else []
        # Missing inputs -> median, plus a 0/1 "was missing" flag the network can use.
        return Pipeline(steps + [("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                 ("scale", StandardScaler())])

    linear = [c for c in spec["numeric"] if c not in spec["log_features"]]
    parts = []
    if spec["log_features"]:
        parts.append(("log_num", numeric_steps(True), spec["log_features"]))
    if linear:
        parts.append(("num", numeric_steps(False), linear))
    if spec["categorical"]:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), spec["categorical"]))
    return ColumnTransformer(parts)


def with_target_scaling(pipeline, spec):
    """Networks train best on a standardised target; skewed targets are log-transformed first."""
    if spec["log_target"]:
        transformer = Pipeline([("log", FunctionTransformer(np.log1p, np.expm1, check_inverse=False)),
                                ("scale", StandardScaler())])
    else:
        transformer = StandardScaler()
    return TransformedTargetRegressor(regressor=pipeline, transformer=transformer)


def ann(spec, seed=config.SEED):
    mlp = MLPRegressor(solver="lbfgs", max_iter=2000, random_state=seed)
    return with_target_scaling(Pipeline([("prep", preprocessor(spec)), ("mlp", mlp)]), spec)


def ensemble(best_ann):
    members = [(f"net{i}", clone(best_ann).set_params(regressor__mlp__random_state=i))
               for i in range(config.ENSEMBLE_SIZE)]
    return VotingRegressor(members, n_jobs=-1)


def ridge(spec):
    return with_target_scaling(Pipeline([("prep", preprocessor(spec)),
                                         ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))]), spec)


def forest(spec):
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=config.SEED, n_jobs=-1)
    return with_target_scaling(Pipeline([("prep", preprocessor(spec)), ("rf", rf)]), spec)


def scores(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=int(len(y)), r2=round(float(r2_score(y, p)), 3),
                rmse=round(float(np.sqrt(mean_squared_error(y, p))), 3),
                mae=round(float(mean_absolute_error(y, p)), 3))


def analytical_prediction(df, target):
    cap = corroded_capacity(df["b_mm"], df["D_mm"], df["cover_mm"], df["bar_dia_mm"], df["n_bars"],
                            df["fck_mpa"], df["corr_pct"], df["L_mm"], config.FY_ASSUMED_MPA)
    return pd.Series(cap["pu_corr_kn"] if target == "load_kn" else cap["residual_pct"], index=df.index)


# ---------------------------------------------------------------- data split

def paper_split(df, seed):
    """Pick whole papers for the test set, so no paper is in both training and test data."""
    papers = np.sort(df["source_id"].unique())
    rng = np.random.default_rng(seed)
    n_test = max(1, round(len(papers) * config.TEST_PAPER_FRACTION))
    return set(rng.choice(papers, size=n_test, replace=False).tolist())


def model_rows(df, spec):
    rows = df[df[spec["target"]].notna()]
    if spec.get("only_consistent_loads"):
        rows = rows[rows["load_consistent"]]
    return rows.dropna(subset=spec["required"])


# ---------------------------------------------------------------- training

def train_one(name, spec, df, test_papers):
    rows = model_rows(df, spec)
    feats = spec["numeric"] + spec["categorical"]
    is_test = rows["source_id"].isin(test_papers)
    train, test = rows[~is_test], rows[is_test]
    Xtr, ytr, gtr = train[feats], train[spec["target"]], train["source_id"]
    Xte, yte = test[feats], test[spec["target"]]
    print(f"\n=== {name}: {len(train)} train rows ({gtr.nunique()} papers), "
          f"{len(test)} test rows ({test['source_id'].nunique()} papers)")

    search = GridSearchCV(ann(spec), PARAM_GRID, cv=GroupKFold(n_splits=config.CV_FOLDS),
                          scoring="neg_root_mean_squared_error", n_jobs=-1, refit=False)
    search.fit(Xtr, ytr, groups=gtr)
    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    i = search.best_index_
    cv_rmse = (-search.cv_results_["mean_test_score"][i], search.cv_results_["std_test_score"][i])
    print(f"    best ANN: {best}  |  CV RMSE {cv_rmse[0]:.2f} ± {cv_rmse[1]:.2f} {spec['unit']}")

    best_ann = ann(spec).set_params(**search.best_params_)
    net = ensemble(best_ann).fit(Xtr, ytr)
    pred = pd.Series(net.predict(Xte), index=test.index)

    result = {
        "target": spec["target"], "unit": spec["unit"], "best_params": {k: str(v) for k, v in best.items()},
        "cv_rmse_mean": round(float(cv_rmse[0]), 3), "cv_rmse_std": round(float(cv_rmse[1]), 3),
        "rows_train": len(train), "rows_test": len(test),
        "papers_train": int(gtr.nunique()), "papers_test": int(test["source_id"].nunique()),
        "test": {"ANN": scores(yte, pred),
                 "Linear (ridge)": scores(yte, ridge(spec).fit(Xtr, ytr).predict(Xte)),
                 "Random forest": scores(yte, forest(spec).fit(Xtr, ytr).predict(Xte))},
    }

    extra = {}
    if spec["target"] in ("load_kn", "residual_pct"):
        ana = analytical_prediction(test, spec["target"]).replace([np.inf, -np.inf], np.nan)
        ok = ana.notna()
        result["head_to_head_rows_with_full_IS456_inputs"] = {
            "ANN": scores(yte[ok], pred[ok]),
            f"IS 456 (fy = {config.FY_ASSUMED_MPA:.0f} MPa)": scores(yte[ok], ana[ok]),
        }
        extra["analytical"] = ana

    # Same network, but rows split at random instead of by paper: shows how much a
    # careless split inflates the score (specimens from one paper are near-duplicates).
    r_tr, r_te = train_test_split(rows, test_size=len(test) / len(rows), random_state=config.SEED)
    leaky = ensemble(best_ann).fit(r_tr[feats], r_tr[spec["target"]])
    result["random_row_split_ANN_for_reference"] = scores(r_te[spec["target"]], leaky.predict(r_te[feats]))

    imp = permutation_importance(net, Xte, yte, n_repeats=15, random_state=config.SEED,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    result["permutation_importance_rmse_increase"] = importance.round(3).to_dict()

    saved = test[["excel_row", "source_id", "specimen"] + feats + [spec["target"]]].copy()
    saved["ann_pred"] = pred
    if "analytical" in extra:
        saved["is456_pred"] = extra["analytical"]
    pred_dir = config.OUT_DIR / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    saved.to_csv(pred_dir / f"{name}.csv", index=False)

    final = ensemble(best_ann).fit(rows[feats], rows[spec["target"]])
    joblib.dump({"model": final, "spec": spec, "features": feats, "best_params": best,
                 "trained_rows": len(rows), "fy_assumed_mpa": config.FY_ASSUMED_MPA},
                config.MODEL_DIR / f"{name}.joblib")

    for method, s in result["test"].items():
        print(f"    test {method:15s} R² {s['r2']:6.3f}  RMSE {s['rmse']:8.2f}  MAE {s['mae']:8.2f}  (n={s['n']})")
    if "head_to_head_rows_with_full_IS456_inputs" in result:
        for method, s in result["head_to_head_rows_with_full_IS456_inputs"].items():
            print(f"    h2h  {method:26s} R² {s['r2']:6.3f}  RMSE {s['rmse']:8.2f}  (n={s['n']})")
    s = result["random_row_split_ANN_for_reference"]
    print(f"    random-row split (leaky) ANN R² {s['r2']:.3f}  RMSE {s['rmse']:.2f}")
    return result, dict(test=test, pred=pred, importance=importance, net=net, train=train, **extra)


# ---------------------------------------------------------------- figures

def parity_plot(name, spec, res, out):
    test, pred = out["test"], out["pred"]
    y = test[spec["target"]]
    fig, ax = P.plt.subplots(figsize=(5.2, 5))
    P.points(ax, y, pred, P.BLUE, label=f"ANN (R² {res['test']['ANN']['r2']:.2f})")
    if "analytical" in out:
        ana = out["analytical"].dropna()
        h2h = res["head_to_head_rows_with_full_IS456_inputs"]
        r2 = [v for k, v in h2h.items() if k.startswith("IS 456")][0]["r2"]
        P.points(ax, y[ana.index], ana, P.ORANGE, label=f"IS 456 calculation (R² {r2:.2f})", alpha=0.55)
    vals = np.concatenate([y.values, pred.values])
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    if spec["target"] == "load_kn":  # loads span 3-1000 kN; corrosion has true zeros, so stays linear
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo = max(lo, 0.1)
    ax.plot([lo, hi], [lo, hi], color=P.MUTED, lw=1, zorder=2)
    ax.set_xlabel(f"Measured {spec['label'].lower()} ({spec['unit']})")
    ax.set_ylabel(f"Predicted ({spec['unit']})")
    ax.set_title(f"{spec['label']}: predicted vs measured\n", loc="left")
    ax.text(0, 1.02, "Held-out papers only - none of these specimens were used in training",
            transform=ax.transAxes, color=P.INK_2, fontsize=8.5)
    ax.legend(loc="lower right")
    P.save(fig, config.FIG_DIR / f"parity_{name}.png")


FEATURE_LABELS = {
    "corr_pct": "Corrosion (%)", "fck_mpa": "Concrete strength fck", "b_mm": "Width b", "D_mm": "Overall depth D",
    "L_mm": "Length L", "cover_mm": "Cover", "bar_dia_mm": "Bar diameter", "n_bars": "No. of tension bars",
    "fiber_pct": "Fibre content", "ast_mm2": "Steel area Ast", "d_eff_mm": "Effective depth d",
    "rho_pct": "Steel ratio ρ", "L_over_D": "Length / depth", "exposure_days": "Exposure duration",
    "nacl_pct": "NaCl concentration", "half_cell_mv": "Half-cell potential", "cl_rebar_pct": "Chloride at rebar",
    "wc_ratio": "w/c ratio", "pre_crack_mm": "Pre-crack width", "method": "Corrosion method",
}


def importance_plot(name, spec, imp):
    imp = imp[imp > 0].sort_values()
    if imp.empty:
        return
    fig, ax = P.plt.subplots(figsize=(5.5, 0.32 * len(imp) + 1.2))
    ax.barh([FEATURE_LABELS.get(f, f) for f in imp.index], imp.values, color=P.BLUE, height=0.55)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Increase in test RMSE when the input is shuffled ({spec['unit']})")
    ax.set_title(f"Which inputs drive the {spec['label'].lower()} model", loc="left")
    P.save(fig, config.FIG_DIR / f"importance_{name}.png")


def typical_beam(beams):
    """Median section of the beams that have a matched control (a realistic lab beam)."""
    b = beams[beams["residual_pct"].notna()]
    t = b[["b_mm", "D_mm", "L_mm", "cover_mm", "bar_dia_mm", "n_bars", "fck_mpa"]].median().round()
    t["fiber_pct"] = 0.0
    t["ast_mm2"] = t["n_bars"] * np.pi * t["bar_dia_mm"] ** 2 / 4
    t["d_eff_mm"] = t["D_mm"] - t["cover_mm"] - t["bar_dia_mm"] / 2
    t["rho_pct"] = 100 * t["ast_mm2"] / (t["b_mm"] * t["d_eff_mm"])
    t["L_over_D"] = t["L_mm"] / t["D_mm"]
    return t


def residual_curve_plot(beams, m3_model, typ):
    corr = np.linspace(0, 40, 81)
    X = pd.DataFrame([typ.to_dict()] * len(corr))
    X["corr_pct"] = corr
    ann_curve = m3_model.predict(X[SPECS["M3_beam_residual_strength"]["numeric"]])
    ana = corroded_capacity(typ["b_mm"], typ["D_mm"], typ["cover_mm"], typ["bar_dia_mm"], typ["n_bars"],
                            typ["fck_mpa"], corr, typ["L_mm"], config.FY_ASSUMED_MPA)["residual_pct"]
    data = beams.dropna(subset=["corr_pct", "residual_pct"])
    data = data[data["corr_pct"] <= 45]
    bins = pd.cut(data["corr_pct"], np.arange(-0.01, 46, 5))
    means = data.groupby(bins, observed=True).agg(c=("corr_pct", "mean"), r=("residual_pct", "mean"),
                                                  n=("residual_pct", "size"))
    means = means[means["n"] >= 5]

    fig, ax = P.plt.subplots(figsize=(7.5, 4.8))
    P.points(ax, data["corr_pct"], data["residual_pct"], P.BLUE, label="Experimental beams", size=18, alpha=0.35)
    ax.scatter(means["c"], means["r"], s=46, color=P.INK, marker="D", zorder=4, edgecolors=P.SURFACE,
               linewidths=1, label="Experimental mean (5 % bins)")
    ax.plot(corr, ann_curve, color=P.ORANGE, label="ANN (M3)", zorder=5)
    ax.plot(corr, ana, color=P.AQUA, label="IS 456 (steel-area loss only)", zorder=5)
    ax.set_xlim(-1, 41)
    ax.set_ylim(30, 140)
    ax.set_xlabel("Average rebar corrosion (% mass loss)")
    ax.set_ylabel("Residual strength (% of uncorroded control)")
    ax.set_title("Residual flexural strength vs corrosion\n", loc="left")
    desc = (f"Curves for a typical beam: {typ['b_mm']:.0f}×{typ['D_mm']:.0f} mm, span {typ['L_mm']:.0f} mm, "
            f"{typ['n_bars']:.0f}Ø{typ['bar_dia_mm']:.0f}, fck {typ['fck_mpa']:.0f} MPa")
    ax.text(0, 1.02, desc, transform=ax.transAxes, color=P.INK_2, fontsize=8.5)
    ax.legend(loc="lower left", ncols=2)
    P.save(fig, config.FIG_DIR / "residual_strength_vs_corrosion.png")


def rebar_facets_plot(rebars, models):
    methods = ["Impressed current", "Salt spray / Environmental", "Natural", "Other / Unspecified"]
    corr = np.linspace(0, 60, 61)
    fig, axes = P.plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
    for row, (name, title) in enumerate([("M4a_rebar_residual_yield", "Residual yield stress (%)"),
                                         ("M4b_rebar_residual_ultimate", "Residual ultimate stress (%)")]):
        target = SPECS[name]["target"]
        for col, method in enumerate(methods):
            ax = axes[row, col]
            sub = rebars[(rebars["method"].isin([method, "Control"])) & rebars[target].notna()]
            P.points(ax, sub["corr_pct"], sub[target], P.BLUE, size=12, alpha=0.35,
                     label="Tests (incl. uncorroded controls)")
            own = rebars[(rebars["method"] == method) & rebars[target].notna()]
            dia = own["bar_dia_mm"].median()
            c = corr[corr <= own["corr_pct"].quantile(0.98)]  # draw the ANN only where this method has data
            X = pd.DataFrame({"corr_pct": c, "bar_dia_mm": dia, "method": np.where(c <= 0.5, "Control", method)})
            ax.plot(c, models[name].predict(X), color=P.ORANGE, label=f"ANN, Ø{dia:.0f} mm bar")
            ax.set_ylim(0, 140)
            if row == 0:
                ax.set_title(method, fontsize=10)
            if col == 0:
                ax.set_ylabel(title)
            if row == 1:
                ax.set_xlabel("Average corrosion (%)")
            ax.legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Corroded rebar: residual strength by corrosion method", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    P.save(fig, config.FIG_DIR / "rebar_residual_by_method.png")


# ---------------------------------------------------------------- main

def performance_table(metrics):
    lines = ["| Model | Train / test rows (papers) | Method | R² | RMSE | MAE |", "|---|---|---|---|---|---|"]
    for name, m in metrics["models"].items():
        rows = f"{m['rows_train']} / {m['rows_test']} ({m['papers_train']} / {m['papers_test']})"
        for method, s in m["test"].items():
            lines.append(f"| {name} | {rows} | {method} | {s['r2']:.2f} | {s['rmse']:.2f} {m['unit']} "
                         f"| {s['mae']:.2f} {m['unit']} |")
            rows = ""
    return "\n".join(lines)


def main():
    for d in (config.MODEL_DIR, config.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    P.apply_style()
    data = {"beam": pd.read_csv(config.BEAM_CLEAN_CSV), "rebar": pd.read_csv(config.REBAR_CLEAN_CSV)}
    split = {k: paper_split(v, config.SEED) for k, v in data.items()}
    (config.OUT_DIR / "split.json").write_text(json.dumps({k: sorted(v) for k, v in split.items()}, indent=1))

    if sys.argv[1:] == ["--figures"]:
        return redraw_figures(data)
    names = sys.argv[1:] or list(SPECS)
    unknown = set(names) - set(SPECS)
    if unknown:
        raise SystemExit(f"Unknown model(s) {sorted(unknown)}; choose from {list(SPECS)}")
    metrics = {"models": {}}
    if names != list(SPECS) and config.METRICS_JSON.exists():
        metrics = json.loads(config.METRICS_JSON.read_text())  # keep results of models not retrained
    metrics.update(fy_assumed_mpa=config.FY_ASSUMED_MPA, ensemble_size=config.ENSEMBLE_SIZE)
    for name in names:
        spec = SPECS[name]
        res, out = train_one(name, spec, data[spec["dataset"]], split[spec["dataset"]])
        metrics["models"][name] = res
        parity_plot(name, spec, res, out)
        importance_plot(name, spec, out["importance"])
    metrics["models"] = {n: metrics["models"][n] for n in SPECS if n in metrics["models"]}

    finals = {n: joblib.load(config.MODEL_DIR / f"{n}.joblib")["model"] for n in SPECS}
    typ = typical_beam(data["beam"])
    metrics["typical_beam"] = typ.round(2).to_dict()
    residual_curve_plot(data["beam"], finals["M3_beam_residual_strength"], typ)
    rebar_facets_plot(data["rebar"], finals)

    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    table = performance_table(metrics)
    (config.OUT_DIR / "model_performance.md").write_text(
        "# Test-set performance (held-out papers)\n\n" + table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nSaved models to {config.MODEL_DIR}, figures to {config.FIG_DIR}, metrics to {config.METRICS_JSON}")


def redraw_figures(data):
    """Rebuild every figure from saved metrics, test predictions and models - no training."""
    metrics = json.loads(config.METRICS_JSON.read_text())
    for name, res in metrics["models"].items():
        spec = SPECS[name]
        test = pd.read_csv(config.OUT_DIR / "test_predictions" / f"{name}.csv")
        out = {"test": test, "pred": test["ann_pred"]}
        if "is456_pred" in test:
            out["analytical"] = test["is456_pred"]
        parity_plot(name, spec, res, out)
        importance_plot(name, spec, pd.Series(res["permutation_importance_rmse_increase"]))
    finals = {n: joblib.load(config.MODEL_DIR / f"{n}.joblib")["model"] for n in SPECS}
    residual_curve_plot(data["beam"], finals["M3_beam_residual_strength"], typical_beam(data["beam"]))
    rebar_facets_plot(data["rebar"], finals)
    print(f"Figures redrawn in {config.FIG_DIR}")


if __name__ == "__main__":
    main()
