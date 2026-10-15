"""Step 3 - Compare experiment vs ANN vs IS 456 (and ABAQUS, once available).

Uses only the held-out test papers, so the ANN is judged on beams it never saw.

Writes:
  outputs/comparison_by_corrosion_level.csv / .md   mean residual strength per corrosion band
  outputs/figures/comparison_by_corrosion_level.png
  outputs/abaqus_benchmark_beams.csv   beams for the team to model in ABAQUS (objective 3).
      Fill in the two 'abaqus_*' columns and re-run this script: it then adds ABAQUS
      to the comparison (objective 4). An existing file is never overwritten.

Run:  python 03_compare_analytical.py      (after 02_train_models.py)
"""
import numpy as np
import pandas as pd

import config
import plotting as P

PRED_DIR = config.OUT_DIR / "test_predictions"
BANDS = [-0.01, 0.5, 5, 10, 15, 20, 30, 80]
BAND_LABELS = ["0 (control)", "0.5-5", "5-10", "10-15", "15-20", "20-30", ">30"]
ABAQUS_COLS = ["abaqus_load_kn", "abaqus_residual_pct"]


def load_test_predictions():
    """Test beams with a residual strength (M3), plus the M2 failure-load predictions for the same rows."""
    m2 = pd.read_csv(PRED_DIR / "M2_beam_failure_load.csv")
    m3 = pd.read_csv(PRED_DIR / "M3_beam_residual_strength.csv")
    m3 = m3.rename(columns={"ann_pred": "ann_residual_pct", "is456_pred": "is456_residual_pct"})
    m2 = m2[["excel_row", "load_kn", "ann_pred", "is456_pred"]].rename(
        columns={"ann_pred": "ann_load_kn", "is456_pred": "is456_load_kn"})
    return m3.merge(m2, on="excel_row", how="left")


def band_table(df):
    d = df.dropna(subset=["residual_pct", "ann_residual_pct", "is456_residual_pct", "corr_pct"]).copy()
    d["band"] = pd.cut(d["corr_pct"], BANDS, labels=BAND_LABELS)
    rows = []
    for band, g in d.groupby("band", observed=True):
        rows.append({
            "corrosion band (%)": band, "beams": len(g),
            "measured mean (%)": g["residual_pct"].mean(),
            "ANN mean (%)": g["ann_residual_pct"].mean(),
            "IS 456 mean (%)": g["is456_residual_pct"].mean(),
            "ANN MAE (%)": (g["ann_residual_pct"] - g["residual_pct"]).abs().mean(),
            "IS 456 MAE (%)": (g["is456_residual_pct"] - g["residual_pct"]).abs().mean(),
        })
    return pd.DataFrame(rows).round(1)


def band_plot(table):
    x = np.arange(len(table))
    fig, ax = P.plt.subplots(figsize=(7.5, 4.4))
    series = [("measured mean (%)", P.BLUE, "Measured (experiments)"),
              ("ANN mean (%)", P.ORANGE, "ANN (M3)"),
              ("IS 456 mean (%)", P.AQUA, "IS 456 (steel-area loss only)")]
    for k, (col, color, label) in enumerate(series):
        ax.plot(x + (k - 1) * 0.12, table[col], color=color, lw=1, alpha=0.5, zorder=2)
        P.points(ax, x + (k - 1) * 0.12, table[col], color, label=label, size=48, alpha=1)
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(table["corrosion band (%)"], table["beams"])])
    ax.set_xlabel("Average corrosion band (%)")
    ax.set_ylabel("Mean residual strength (%)")
    ax.set_title("Residual strength by corrosion level - held-out test beams\n", loc="left")
    ax.text(0, 1.02, "Means per band; the ANN never saw these papers during training",
            transform=ax.transAxes, color=P.INK_2, fontsize=8.5)
    ax.legend(loc="lower left")
    P.save(fig, config.FIG_DIR / "comparison_by_corrosion_level.png")


def pick_benchmark_beams(df, max_papers=4):
    """Complete test-set series (a control plus at least two corroded beams) for ABAQUS."""
    need = ["b_mm", "D_mm", "L_mm", "cover_mm", "bar_dia_mm", "n_bars", "fck_mpa", "corr_pct", "load_kn",
            "residual_pct", "ann_load_kn", "ann_residual_pct", "is456_load_kn"]
    d = df.dropna(subset=need)
    good = []
    for sid, g in d.groupby("source_id"):
        if (g["corr_pct"] <= 0.5).any() and (g["corr_pct"] > 0.5).sum() >= 2:
            good.append((g["corr_pct"].max(), sid))
    chosen = [sid for _, sid in sorted(good, reverse=True)[:max_papers]]  # widest corrosion range first
    cols = ["source_id", "excel_row", "specimen", "b_mm", "D_mm", "L_mm", "cover_mm", "bar_dia_mm", "n_bars",
            "fck_mpa", "fiber_pct", "corr_pct", "load_kn", "residual_pct", "ann_load_kn", "ann_residual_pct",
            "is456_load_kn", "is456_residual_pct"]
    out = d[d["source_id"].isin(chosen)].sort_values(["source_id", "corr_pct"])[cols].round(2)
    for c in ABAQUS_COLS:
        out[c] = np.nan
    return out


def abaqus_comparison(bench):
    done = bench.dropna(subset=["abaqus_load_kn"])
    if done.empty:
        print(f"\nABAQUS: no results yet. Fill 'abaqus_load_kn' (and optionally 'abaqus_residual_pct') in\n"
              f"  {config.ABAQUS_TEMPLATE_CSV}\nthen re-run this script.")
        return
    if done["abaqus_residual_pct"].isna().all():
        ctrl = done[done["corr_pct"] <= 0.5].groupby("source_id")["abaqus_load_kn"].mean()
        done = done.assign(abaqus_residual_pct=100 * done["abaqus_load_kn"] / done["source_id"].map(ctrl))
    rows = []
    for label, col_load, col_res in [("ANN", "ann_load_kn", "ann_residual_pct"),
                                     ("IS 456", "is456_load_kn", "is456_residual_pct"),
                                     ("ABAQUS", "abaqus_load_kn", "abaqus_residual_pct")]:
        e_load = (done[col_load] - done["load_kn"]) / done["load_kn"] * 100
        e_res = done[col_res] - done["residual_pct"]
        rows.append({"method": label, "beams": len(done),
                     "load: mean error (%)": e_load.mean(), "load: mean |error| (%)": e_load.abs().mean(),
                     "residual: MAE (% points)": e_res.abs().mean()})
    table = pd.DataFrame(rows).round(1)
    table.to_csv(config.OUT_DIR / "comparison_with_abaqus.csv", index=False)
    print("\nFour-way comparison on the ABAQUS benchmark beams:")
    print(table.to_string(index=False))


def main():
    P.apply_style()
    df = load_test_predictions()

    table = band_table(df)
    table.to_csv(config.OUT_DIR / "comparison_by_corrosion_level.csv", index=False)
    (config.OUT_DIR / "comparison_by_corrosion_level.md").write_text(
        "# Residual strength by corrosion level (held-out test beams)\n\n" + table.to_markdown(index=False) + "\n",
        encoding="utf-8")
    band_plot(table)
    print("Residual strength by corrosion band (held-out test beams):")
    print(table.to_string(index=False))

    if config.ABAQUS_TEMPLATE_CSV.exists():
        bench = pd.read_csv(config.ABAQUS_TEMPLATE_CSV)
        print(f"\nUsing existing {config.ABAQUS_TEMPLATE_CSV.name} ({len(bench)} beams); delete it to regenerate.")
    else:
        bench = pick_benchmark_beams(df)
        bench.to_csv(config.ABAQUS_TEMPLATE_CSV, index=False)
        print(f"\nWrote ABAQUS benchmark template with {len(bench)} beams from "
              f"{bench['source_id'].nunique()} papers -> {config.ABAQUS_TEMPLATE_CSV}")
    abaqus_comparison(bench)


if __name__ == "__main__":
    main()
