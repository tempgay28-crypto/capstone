"""Step 1 - Clean the beam and rebar spreadsheets into model-ready CSV files.

Reads the original Excel files (never modifies them) and writes:
  outputs/data/beam_clean.csv     one row per corroded / control RC beam
  outputs/data/rebar_clean.csv    one row per rebar tensile test
  outputs/data/cleaning_log.csv   every value that was changed or removed, with the
                                  Excel row number, so the source sheets can be fixed

Run:  python 01_clean_data.py
"""
import re

import numpy as np
import pandas as pd

import config
from webapp.rcassess.analytical import corroded_capacity, effective_depth, steel_area

LOG = []


def log(dataset, excel_row, column, raw, action, reason):
    LOG.append(dict(dataset=dataset, excel_row=excel_row, column=column,
                    raw_value=raw, action=action, reason=reason))


# ---------------------------------------------------------------- parsing helpers

_NUM = r"-?\d+(?:\.\d+)?"
_POS = r"\d+(?:\.\d+)?"


def parse_number(v):
    """Turn a spreadsheet cell into a float, or NaN if it is not a clear number.

    Accepts notes and symbols around a number ('5% (sprayed)', '~29.7', '10*') and
    ranges ('25-30' -> 27.5). Anything else ('Level 2', 'Variable', dates) -> NaN.
    """
    if v is None:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if not isinstance(v, str):
        return np.nan  # e.g. a number Excel auto-converted to a date
    s = re.sub(r"\(.*?\)", " ", v)
    s = re.sub(r"\(.*$", " ", s)
    s = s.replace("~", "").replace("≈", "").replace("*", "").replace("%", "").strip()
    if not s:
        return np.nan
    if re.fullmatch(_NUM, s):
        return float(s)
    m = re.fullmatch(rf"({_NUM})\s*[-–—]\s*({_POS})", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    return np.nan


def is_blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())


def as_text(series):
    """Blank cells -> '' so string matching never sees NaN (pandas 3 keeps NaN after astype(str))."""
    return series.map(lambda v: "" if is_blank(v) else str(v).strip())


def norm_header(h):
    return re.sub(r"\s+", " ", str(h)).strip().lower()


def pick_columns(df, mapping, dataset):
    """Rename columns by header prefix so the script survives column re-ordering."""
    headers = {norm_header(c): c for c in df.columns}
    rename = {}
    for prefix, short in mapping.items():
        hits = [orig for h, orig in headers.items() if h.startswith(prefix)]
        if len(hits) != 1:
            raise KeyError(f"{dataset}: expected one column starting with '{prefix}', found {hits}")
        rename[hits[0]] = short
    return df[list(rename)].rename(columns=rename)


def to_numeric_logged(df, cols, dataset, row_offset):
    for col in cols:
        raw = df[col]
        parsed = raw.map(parse_number).astype(float)
        bad = parsed.isna() & ~raw.map(is_blank)
        for i in df.index[bad]:
            log(dataset, i + row_offset, col, raw[i], "set blank", "not a number")
        df[col] = parsed
    return df


def apply_bounds(df, bounds, dataset, row_offset):
    """Blank out physically implausible values (most are typing or column-shift errors)."""
    for col, (lo, hi) in bounds.items():
        out = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        for i in df.index[out]:
            log(dataset, i + row_offset, col, df.at[i, col], "set blank", f"outside plausible range [{lo}, {hi}]")
        df.loc[out, col] = np.nan
    return df


def source_groups(primary, secondary):
    """Each paper's reference is written only on its first row; carry it down."""
    key = primary.where(~primary.map(is_blank), secondary)
    key = key.where(~key.map(is_blank)).ffill().fillna("unknown")
    key = key.map(lambda s: re.sub(r"\s+", " ", str(s)).strip().lower())
    return pd.factorize(key)[0]


# ---------------------------------------------------------------- beams

BEAM_COLUMNS = {
    "specimen id": "specimen",
    "type": "type",
    "cement type": "cement",
    "concrete grade": "fck_mpa",
    "w/c ratio": "wc_ratio",
    "fiber": "fiber_pct",
    "exposure nacl": "nacl_pct",
    "potential difference": "half_cell_mv",
    "beam width": "b_mm",
    "beam depth": "D_mm",
    "beam length": "L_mm",
    "cover": "cover_mm",
    "bar dia": "bar_dia_mm",
    "nos. tension": "n_bars",
    "exposure (days)": "exposure_days",
    "cl at rebar": "cl_rebar_pct",
    "average corrosion": "corr_pct",
    "load at failure": "load_kn",
    "pre-exposure crack width": "pre_crack_mm",
    "research paper": "paper",
    "citation": "citation",
}

BEAM_NUMERIC = ["fck_mpa", "wc_ratio", "fiber_pct", "nacl_pct", "half_cell_mv", "b_mm", "D_mm",
                "L_mm", "cover_mm", "bar_dia_mm", "n_bars", "exposure_days", "cl_rebar_pct",
                "corr_pct", "load_kn", "pre_crack_mm"]

BEAM_BOUNDS = {
    "fck_mpa": (10, 120), "wc_ratio": (0.25, 0.8), "fiber_pct": (0, 4), "nacl_pct": (0, 10),
    "half_cell_mv": (-800, 200), "b_mm": (50, 1000), "D_mm": (75, 1500), "L_mm": (300, 12000),
    "cover_mm": (5, 100), "bar_dia_mm": (6, 40), "n_bars": (1, 16), "exposure_days": (0, 20000),
    "cl_rebar_pct": (0, 10), "corr_pct": (0, 80), "load_kn": (0.5, 5000), "pre_crack_mm": (0, 5),
}

# Plain RC beams in bending. Shear-critical, strengthened, prestressed and continuous
# members fail differently and would confuse a flexural-capacity model.
BEAM_TYPES_KEPT = {"beam", "rc beam", "rc beam (rac)"}

CONTROL_MAX_CORR = 0.5  # % mass loss treated as "uncorroded control"


def clean_beams():
    ds, offset = "beam", 2  # Excel row = dataframe index + 2 (one header row)
    raw = pd.read_excel(config.BEAM_XLSX, header=0, dtype=object)
    raw = raw.dropna(how="all")
    df = pick_columns(raw, BEAM_COLUMNS, ds)
    df["source_id"] = source_groups(df["paper"], df["citation"])
    df["excel_row"] = df.index + offset

    df["type"] = as_text(df["type"])
    drop = ~df["type"].str.lower().isin(BEAM_TYPES_KEPT)
    for i in df.index[drop]:
        log(ds, i + offset, "type", df.at[i, "type"], "row dropped", "not a plain RC flexural beam")
    df = df[~drop].copy()

    df = to_numeric_logged(df, BEAM_NUMERIC, ds, offset)
    df = apply_bounds(df, BEAM_BOUNDS, ds, offset)

    bad_cover = df["cover_mm"].notna() & df["D_mm"].notna() & (df["cover_mm"] >= df["D_mm"] / 2)
    for i in df.index[bad_cover]:
        log(ds, i + offset, "cover_mm", df.at[i, "cover_mm"], "set blank", "cover >= half the beam depth")
    df.loc[bad_cover, "cover_mm"] = np.nan

    # The same specimen entered twice (often by two people, under different paper
    # labels) would leak between training and test sets.
    dup_key = ["b_mm", "D_mm", "L_mm", "cover_mm", "bar_dia_mm", "n_bars", "fck_mpa", "corr_pct", "load_kn"]
    has_key = df[["b_mm", "D_mm", "corr_pct", "load_kn"]].notna().all(axis=1)
    dup = has_key & df.duplicated(subset=dup_key, keep="first")
    for i in df.index[dup]:
        log(ds, i + offset, "*", df.at[i, "specimen"], "row dropped", "duplicate of an earlier row")
    df = df[~dup].copy()

    df["cement"] = as_text(df["cement"]).str.upper()
    df["is_ppc"] = df["cement"].str.contains("PPC").astype(float)

    # Derived section properties (same definitions as the IS 456 calculation sheet)
    df["ast_mm2"] = steel_area(df["bar_dia_mm"], df["n_bars"])
    df["d_eff_mm"] = effective_depth(df["D_mm"], df["cover_mm"], df["bar_dia_mm"])
    df["rho_pct"] = 100 * df["ast_mm2"] / (df["b_mm"] * df["d_eff_mm"])
    df["L_over_D"] = df["L_mm"] / df["D_mm"]

    df = add_residual_strength(df, ds, offset)
    df = check_load_against_is456(df, ds, offset)
    cols = ["excel_row", "source_id", "specimen", "type", "cement", "is_ppc"] + BEAM_NUMERIC + \
           ["ast_mm2", "d_eff_mm", "rho_pct", "L_over_D", "control_load_kn", "residual_pct",
            "load_to_is456", "load_consistent"]
    return df[cols]


def check_load_against_is456(df, ds, offset):
    """Flag failure loads that no flexural failure of the recorded section could produce.

    Ratio = measured load / IS 456 design capacity at the measured corrosion. Design
    factors, the unknown fy and 3- vs 4-point loading explain ratios up to ~3.5;
    far outside config.LOAD_TO_IS456_VALID the row almost certainly has a unit,
    dimension or member-type error. Such rows are kept (their residual % is still
    valid, because it is relative to the same paper's control) but are excluded from
    the absolute failure-load model.
    """
    cap = corroded_capacity(df["b_mm"], df["D_mm"], df["cover_mm"], df["bar_dia_mm"], df["n_bars"],
                            df["fck_mpa"], df["corr_pct"], df["L_mm"], config.FY_ASSUMED_MPA)
    df["load_to_is456"] = df["load_kn"] / cap["pu_corr_kn"]
    lo, hi = config.LOAD_TO_IS456_VALID
    out = df["load_to_is456"].notna() & ((df["load_to_is456"] < lo) | (df["load_to_is456"] > hi))
    for i in df.index[out]:
        log(ds, i + offset, "load_kn", f"{df.at[i, 'load_kn']:g} kN = {df.at[i, 'load_to_is456']:.1f}x IS 456",
            "excluded from load model", f"load outside {lo}-{hi}x the IS 456 capacity of the recorded section"
            " - check units, dimensions, bars or member type")
    df["load_consistent"] = ~out
    return df


def add_residual_strength(df, ds, offset):
    """Residual strength % = load / load of the matching uncorroded control x 100.

    The spreadsheet's own 'Reduction in Load Capacity (%)' column is not used: in
    999 of 1005 rows it is a copy of 'Max Bending Moment'. A control must come from
    the same paper and have identical section, span, reinforcement and concrete.
    """
    match_cols = ["source_id", "b_mm", "D_mm", "L_mm", "cover_mm", "bar_dia_mm", "n_bars", "fck_mpa", "fiber_pct"]
    key = df[match_cols].round(3).fillna(-1).astype(str).agg("|".join, axis=1)
    is_ctrl = (df["corr_pct"] <= CONTROL_MAX_CORR) & df["load_kn"].notna()
    ctrl_load = df[is_ctrl].groupby(key[is_ctrl])["load_kn"].mean()
    df["control_load_kn"] = key.map(ctrl_load)
    df["residual_pct"] = 100 * df["load_kn"] / df["control_load_kn"]

    lo, hi = config.RESIDUAL_PCT_VALID
    out = df["residual_pct"].notna() & ((df["residual_pct"] < lo) | (df["residual_pct"] > hi))
    for i in df.index[out]:
        log(ds, i + offset, "residual_pct", round(df.at[i, "residual_pct"], 1), "set blank",
            f"residual strength outside [{lo}, {hi}] % - check load or control")
    df.loc[out, "residual_pct"] = np.nan
    return df


# ---------------------------------------------------------------- rebars

REBAR_COLUMNS = {
    "citation": "citation",
    "doi": "doi",
    "exposure days": "exposure_days",
    "solution concentration": "nacl_pct",
    "specimen name": "specimen",
    "bar nominal diameter": "bar_dia_mm",
    "corrosion current density": "icorr_ua_cm2",
    "avg % corrosion": "corr_pct",
    "residual engg. yield stress": "ry_eng_pct",
    "residual eff. yield stress": "ry_eff_pct",
    "engg. nominal ultimate stress": "ru_eng_pct",
    "residual eff. ultimate stress": "ru_eff_pct",
    "corrosion method": "method_raw",
    "conditions": "conditions",
}

REBAR_NUMERIC = ["exposure_days", "nacl_pct", "bar_dia_mm", "icorr_ua_cm2", "corr_pct",
                 "ry_eng_pct", "ry_eff_pct", "ru_eng_pct", "ru_eff_pct"]

REBAR_BOUNDS = {
    "exposure_days": (0, 40000), "nacl_pct": (0, 20), "bar_dia_mm": (4, 50), "icorr_ua_cm2": (0, 5000),
    "corr_pct": (0, 90), "ry_eng_pct": (5, 150), "ru_eng_pct": (5, 150),
    "ry_eff_pct": (20, 200), "ru_eff_pct": (20, 200),
}

# Same five groups as the literature-review figure in the progress presentation.
METHOD_KEYWORDS = [
    ("Impressed current", ["impressed", "electrochem", "galvanostatic", "accelerated", "dc power", "(ic)"]),
    ("Salt spray / Environmental", ["salt spray", "environment", "artificial climate", "wet-dry", "wet/dry",
                                    "alternate immersion", "chamber", "(ei)", "cyclic immersion"]),
    ("Natural", ["natural", "in service", "existing", "carbonation", "marine", "field"]),
]
CONTROL_WORDS = ["control", "uncorroded", "non-corroded", "virgin", "as-received", "reference"]
NON_REBAR_WORDS = ["bolt", "strand", "wire"]


def rebar_source_ids(citation, doi):
    """Paper key = DOI. One DOI (a Zenodo compilation) bundles hundreds of older studies,
    so inside any DOI shared by several 'Author (year)' citations, use author + year."""
    def norm_doi(v):
        m = re.search(r"10\.\d{4,9}/\S+", v)
        return m.group(0).lower().rstrip(".") if m else (v.lower() or None)

    def author_year(v):
        m = re.match(r"^(.*?\(\d{4}\))", v)
        return re.sub(r"\s+", " ", m.group(1)).strip().lower() if m else None

    doi_key = as_text(doi).map(norm_doi)
    ay = as_text(citation).map(author_year)
    n_studies = ay.groupby(doi_key).nunique()
    compilations = set(n_studies[n_studies > 3].index)
    key = [f"{d} | {a}" if d in compilations and a else (d or a or c.lower())
           for d, a, c in zip(doi_key, ay, as_text(citation))]
    return pd.factorize(pd.Series(key, index=citation.index))[0]


def classify_method(text, corr):
    t = str(text).lower()
    if pd.notna(corr) and corr <= CONTROL_MAX_CORR:
        return "Control"
    for label, words in METHOD_KEYWORDS:
        if any(w in t for w in words):
            return label
    return "Other / Unspecified"


def clean_rebars():
    ds, offset = "rebar", 4  # three header rows
    raw = pd.read_excel(config.REBAR_XLSX, header=2, dtype=object)
    raw = raw.dropna(how="all")
    df = pick_columns(raw, REBAR_COLUMNS, ds)
    df["source_id"] = rebar_source_ids(df["citation"], df["doi"])
    df["excel_row"] = df.index + offset

    df["method_raw"] = as_text(df["method_raw"])
    text = (df["method_raw"] + " " + as_text(df["conditions"]) + " " + as_text(df["specimen"])).str.lower()
    drop = text.map(lambda t: any(w in t for w in NON_REBAR_WORDS))
    for i in df.index[drop]:
        log(ds, i + offset, "corrosion method", df.at[i, "method_raw"], "row dropped", "bolt/strand/wire, not a rebar")
    df = df[~drop].copy()

    df = to_numeric_logged(df, REBAR_NUMERIC, ds, offset)

    # Small negative mass loss is weighing noise on uncorroded bars.
    neg = df["corr_pct"].between(-5, 0, inclusive="left")
    for i in df.index[neg]:
        log(ds, i + offset, "corr_pct", df.at[i, "corr_pct"], "set to 0", "small negative mass loss (weighing noise)")
    df.loc[neg, "corr_pct"] = 0.0
    df = apply_bounds(df, REBAR_BOUNDS, ds, offset)

    # A bar cannot lose far more strength than steel: a residual stress below 30 % of the
    # remaining steel fraction means another quantity (elongation, strain) is in the column -
    # e.g. Hou et al. (2019), Berrocal et al., Hansapinyo et al. show 5-18 % at ~0 % corrosion.
    for col in ["ry_eng_pct", "ru_eng_pct"]:
        bad = df[col].notna() & df["corr_pct"].notna() & (df[col] < 0.3 * (100 - df["corr_pct"]))
        for i in df.index[bad]:
            log(ds, i + offset, col, df.at[i, col], "set blank",
                "residual stress < 30 % of the remaining steel area - probably another quantity in this column")
        df.loc[bad, col] = np.nan

    df["method"] = [classify_method(m, c) for m, c in zip(df["method_raw"], df["corr_pct"])]
    labelled_ctrl = df["method_raw"].str.lower().map(lambda t: any(w in t for w in CONTROL_WORDS))
    odd = labelled_ctrl & (df["method"] != "Control")
    for i in df.index[odd]:
        log(ds, i + offset, "corrosion method", df.at[i, "method_raw"], "kept as 'Other'",
            "labelled control but corrosion % missing or > 0.5")

    dup = df.duplicated(subset=["source_id", "specimen", "bar_dia_mm", "corr_pct", "ry_eng_pct", "ru_eng_pct"])
    for i in df.index[dup]:
        log(ds, i + offset, "*", df.at[i, "specimen"], "row dropped", "duplicate of an earlier row")
    df = df[~dup].copy()

    cols = ["excel_row", "source_id", "specimen", "method", "method_raw"] + REBAR_NUMERIC
    return df[cols]


# ---------------------------------------------------------------- main

def summary(beams, rebars, log_df):
    lines = ["CLEANING SUMMARY", "================", ""]
    lines.append(f"Beams kept: {len(beams)} rows from {beams['source_id'].nunique()} papers")
    for col in ["corr_pct", "load_kn", "residual_pct", "half_cell_mv", "exposure_days"]:
        lines.append(f"  {col:15s} available in {beams[col].notna().sum():5d} rows")
    lines.append(f"  beams with a matched control: {beams['control_load_kn'].notna().sum()}")
    lines.append("")
    lines.append(f"Rebars kept: {len(rebars)} rows from {rebars['source_id'].nunique()} papers")
    for col in ["corr_pct", "ry_eng_pct", "ru_eng_pct", "bar_dia_mm"]:
        lines.append(f"  {col:15s} available in {rebars[col].notna().sum():5d} rows")
    lines.append("  corrosion method groups: " + ", ".join(f"{k}={v}" for k, v in rebars["method"].value_counts().items()))
    lines.append("")
    lines.append("Cleaning actions (dataset / action / reason -> count):")
    counts = log_df.groupby(["dataset", "action", "reason"]).size().sort_values(ascending=False)
    for (d, a, r), n in counts.items():
        lines.append(f"  {n:5d}  {d:5s}  {a:15s}  {r}")
    return "\n".join(lines)


def main():
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    beams = clean_beams()
    rebars = clean_rebars()
    log_df = pd.DataFrame(LOG, columns=["dataset", "excel_row", "column", "raw_value", "action", "reason"])

    beams.to_csv(config.BEAM_CLEAN_CSV, index=False)
    rebars.to_csv(config.REBAR_CLEAN_CSV, index=False)
    log_df.to_csv(config.CLEANING_LOG_CSV, index=False)

    text = summary(beams, rebars, log_df)
    (config.CLEAN_DIR / "cleaning_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {config.BEAM_CLEAN_CSV}\n      {config.REBAR_CLEAN_CSV}\n      {config.CLEANING_LOG_CSV}")


if __name__ == "__main__":
    main()
