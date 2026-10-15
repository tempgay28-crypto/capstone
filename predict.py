"""Predict corrosion and residual strength for one beam with the trained ANN models.

Examples
  # corrosion level known (e.g. from a mass-loss test or a core sample)
  python predict.py --b 150 --D 200 --L 1500 --cover 25 --bar-dia 12 --n-bars 2 --fck 30 --corr 12

  # corrosion unknown: opt in to the experimental M1 estimate from exposure and NDT data
  # (M1 scored R² ≈ 0 on unseen studies - see README - so prefer a measured value)
  python predict.py --b 150 --D 200 --L 1500 --cover 25 --bar-dia 12 --n-bars 2 --fck 30 \
                    --estimate-corrosion --exposure-days 180 --nacl 3.5 --half-cell -450 --wc 0.45

  # residual-strength curve from 0 to 40 % corrosion, plus rebar results for a method
  python predict.py --b 150 --D 200 --L 1500 --cover 25 --bar-dia 12 --n-bars 2 --fck 30 --curve \
                    --method "Impressed current"
"""
import argparse
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from analytical import corroded_capacity, effective_depth, steel_area  # noqa: E402

METHODS = ["Impressed current", "Salt spray / Environmental", "Natural", "Other / Unspecified"]


def load(name):
    path = config.MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise SystemExit(f"{path} not found - run 01_clean_data.py and 02_train_models.py first.")
    return joblib.load(path)


def beam_frame(a, corr):
    corr = np.atleast_1d(np.asarray(corr, dtype=float))
    ast = steel_area(a.bar_dia, a.n_bars)
    d = effective_depth(a.D, a.cover, a.bar_dia)
    row = dict(b_mm=a.b, D_mm=a.D, L_mm=a.L, cover_mm=a.cover, bar_dia_mm=a.bar_dia, n_bars=a.n_bars,
               fck_mpa=a.fck, fiber_pct=a.fiber, ast_mm2=ast, d_eff_mm=d, rho_pct=100 * ast / (a.b * d),
               L_over_D=a.L / a.D, wc_ratio=a.wc, exposure_days=a.exposure_days, nacl_pct=a.nacl,
               half_cell_mv=a.half_cell, cl_rebar_pct=a.cl, pre_crack_mm=a.pre_crack)
    df = pd.DataFrame([row] * len(corr))
    df["corr_pct"] = corr
    return df.astype(float)


def predict(bundle, df):
    return bundle["model"].predict(df[bundle["features"]])


def estimate_corrosion(a):
    m1 = load("M1_beam_corrosion")
    est = float(predict(m1, beam_frame(a, np.nan))[0])
    return max(est, 0.0)


def check_range(a):
    """Warn when an input lies outside the range the models were trained on."""
    beams = pd.read_csv(config.BEAM_CLEAN_CSV)
    for col, val in [("b_mm", a.b), ("D_mm", a.D), ("L_mm", a.L), ("cover_mm", a.cover),
                     ("bar_dia_mm", a.bar_dia), ("fck_mpa", a.fck), ("corr_pct", a.corr)]:
        if val is None or np.isnan(val):
            continue
        lo, hi = beams[col].quantile([0.01, 0.99])
        if not lo <= val <= hi:
            print(f"  ! {col} = {val:g} is outside the typical training range {lo:g}-{hi:g}; "
                  f"treat the prediction with caution.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("beam (required)")
    g.add_argument("--b", type=float, required=True, help="beam width, mm")
    g.add_argument("--D", type=float, required=True, help="overall depth, mm")
    g.add_argument("--L", type=float, required=True, help="span / beam length, mm")
    g.add_argument("--cover", type=float, required=True, help="clear cover, mm")
    g.add_argument("--bar-dia", type=float, required=True, help="tension bar diameter, mm")
    g.add_argument("--n-bars", type=float, required=True, help="number of tension bars")
    g.add_argument("--fck", type=float, required=True, help="concrete compressive strength, MPa")
    g.add_argument("--fiber", type=float, default=0.0, help="fibre content, %% (default 0)")
    g.add_argument("--fy", type=float, default=config.FY_ASSUMED_MPA, help="steel yield strength for IS 456, MPa")
    c = p.add_argument_group("corrosion (give a measured --corr; --estimate-corrosion is experimental)")
    c.add_argument("--corr", type=float, help="average corrosion, %% mass loss")
    c.add_argument("--estimate-corrosion", action="store_true",
                   help="estimate corrosion with M1 from the exposure/NDT inputs below (unreliable)")
    c.add_argument("--exposure-days", type=float, default=np.nan)
    c.add_argument("--nacl", type=float, default=np.nan, help="NaCl concentration of exposure, %%")
    c.add_argument("--half-cell", type=float, default=np.nan, help="half-cell potential, mV")
    c.add_argument("--cl", type=float, default=np.nan, help="chloride at rebar level, %%")
    c.add_argument("--wc", type=float, default=np.nan, help="water-cement ratio")
    c.add_argument("--pre-crack", type=float, default=np.nan, help="pre-exposure crack width, mm")
    o = p.add_argument_group("output")
    o.add_argument("--curve", action="store_true", help="print residual strength for 0-40 %% corrosion")
    o.add_argument("--method", choices=METHODS, help="also predict rebar residual stresses for this method")
    a = p.parse_args()

    if a.corr is None and a.curve and not a.estimate_corrosion:
        a.corr = np.nan  # the curve covers 0-40 %, no single corrosion level needed
    elif a.corr is None:
        if not a.estimate_corrosion:
            raise SystemExit("Give a measured --corr (average % mass loss), or use --curve.\n"
                             "Estimating corrosion from exposure/NDT data (M1) is unreliable; add "
                             "--estimate-corrosion with at least --exposure-days to try it anyway.")
        if np.isnan(a.exposure_days):
            raise SystemExit("--estimate-corrosion needs at least --exposure-days (plus any NDT readings).")
        a.corr = estimate_corrosion(a)
        print(f"Estimated corrosion from exposure/NDT data (M1): {a.corr:.1f} %")
        print("  ! M1 is unreliable on unseen studies (test R² ≈ 0): the beam data lack the corrosion method\n"
              "    and current density that actually control corrosion. Prefer a measured --corr.")
    check_range(a)

    m2, m3 = load("M2_beam_failure_load"), load("M3_beam_residual_strength")
    levels = np.arange(0, 41, 5.0) if a.curve else np.array([0.0, a.corr])
    df = beam_frame(a, levels)
    load_kn = predict(m2, df)
    resid_direct = predict(m3, df)
    ana = corroded_capacity(a.b, a.D, a.cover, a.bar_dia, a.n_bars, a.fck, levels, a.L, a.fy)

    out = pd.DataFrame({
        "corrosion %": levels,
        "ANN failure load (kN)": load_kn,
        "ANN residual % (M3)": resid_direct,
        "ANN residual % (M2 ratio)": 100 * load_kn / load_kn[0],
        f"IS 456 Pu (kN, fy={a.fy:g})": ana["pu_corr_kn"],
        "IS 456 residual %": ana["residual_pct"],
    }).round(1)
    if not a.curve:
        out = out.iloc[[1]]
    print()
    print(out.to_string(index=False))
    print("\nNotes: IS 456 Pu uses the design formula (0.87 fy), so it is conservative by design; "
          "Pu = 4 Mu / L assumes a central point load.")

    if a.method:
        m4a, m4b = load("M4a_rebar_residual_yield"), load("M4b_rebar_residual_ultimate")
        rb = pd.DataFrame({"corr_pct": levels, "bar_dia_mm": a.bar_dia,
                           "method": np.where(levels <= 0.5, "Control", a.method)})
        rebar = pd.DataFrame({"corrosion %": levels,
                              "rebar residual yield % (M4a)": predict(m4a, rb),
                              "rebar residual ultimate % (M4b)": predict(m4b, rb)}).round(1)
        if not a.curve:
            rebar = rebar.iloc[[1]]
        print(f"\nRebar ({a.method}, Ø{a.bar_dia:g} mm):")
        print(rebar.to_string(index=False))


if __name__ == "__main__":
    main()
