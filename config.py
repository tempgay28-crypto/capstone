"""Paths and project-wide assumptions. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT.parent / "Capstone Project 2026-27"

BEAM_XLSX = RAW_DIR / "Beam Dataset" / "#Beam dataset 260910.xlsx"
REBAR_XLSX = RAW_DIR / "Rebar Dataset" / "#Rebar spreadsheet.xlsx"

OUT_DIR = ROOT / "outputs"
CLEAN_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = ROOT / "models"

BEAM_CLEAN_CSV = CLEAN_DIR / "beam_clean.csv"
REBAR_CLEAN_CSV = CLEAN_DIR / "rebar_clean.csv"
CLEANING_LOG_CSV = CLEAN_DIR / "cleaning_log.csv"
METRICS_JSON = OUT_DIR / "metrics.json"
ABAQUS_TEMPLATE_CSV = OUT_DIR / "abaqus_benchmark_beams.csv"

SEED = 42

# Fraction of *source papers* held out as the final test set. Splitting by paper
# (not by row) stops near-identical specimens from one study appearing in both
# training and test data, which would make the scores look better than they are.
TEST_PAPER_FRACTION = 0.20
CV_FOLDS = 5

# Number of networks (different random initialisations) averaged in each final model.
ENSEMBLE_SIZE = 5

# The beam dataset does not record steel yield strength, so the IS 456 analytical
# calculation needs an assumed value. Fe500 is the most common grade in the
# source studies and in current Indian practice.
FY_ASSUMED_MPA = 500.0

# Residual strength % = failure load / load of the uncorroded control beam from the
# same paper with the same geometry. Values outside this range are treated as
# data-entry errors.
RESIDUAL_PCT_VALID = (10.0, 160.0)

# Measured failure load / IS 456 design capacity of the recorded section. Outside this
# band the row cannot be a flexural failure of that section (unit, dimension or
# member-type error) and is left out of the absolute failure-load model (M2).
LOAD_TO_IS456_VALID = (0.4, 5.0)
