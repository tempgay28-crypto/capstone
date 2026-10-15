# ANN model – residual strength of corroded RC flexural members

Capstone Project 2026-27, Dept. of Civil Engineering, Walchand College of Engineering, Sangli.
Covers **Objective 2** (ANN to predict corrosion % and residual flexural strength), prepares
**Objective 3** (benchmark beams for ABAQUS) and runs **Objective 4** (comparison of ANN, IS 456
analytical, experimental and – once filled in – ABAQUS results).

The raw spreadsheets in `../Capstone Project 2026-27/` are only read, never modified.

## How to run

```powershell
cd D:\Projects\Civil-ANN-model-2\ann_model
..\.venv\Scripts\python.exe 01_clean_data.py        # ~30 s  -> outputs/data/
..\.venv\Scripts\python.exe 02_train_models.py      # ~30 min on 12 cores -> models/, outputs/figures/, outputs/metrics.json
..\.venv\Scripts\python.exe 03_compare_analytical.py  # ~5 s -> comparison tables + ABAQUS template
..\.venv\Scripts\python.exe predict.py --b 150 --D 200 --L 1500 --cover 25 --bar-dia 12 --n-bars 2 --fck 30 --corr 12
```

Useful variants:

```powershell
..\.venv\Scripts\python.exe 02_train_models.py M3_beam_residual_strength   # retrain one model, keep the others
..\.venv\Scripts\python.exe 02_train_models.py --figures                   # redraw all figures, no training
..\.venv\Scripts\python.exe predict.py ... --curve --method "Impressed current"  # 0-40 % curve + rebar results
```

`predict.py` needs a measured corrosion level (`--corr`). Estimating it from exposure and NDT
data (`--estimate-corrosion`, model M1) is available but unreliable – see Results.

On a fresh machine: `python -m venv .venv` then `.venv\Scripts\python.exe -m pip install -r ann_model\requirements.txt`.

## Web app (stage 1)

```powershell
..\.venv\Scripts\python.exe app.py          # opens http://localhost:8000 (this computer only)
```

`http://localhost:8000/?example` loads and assesses the worked example; add `&theme=light` or
`&theme=dark` to force a theme. **Print** gives a light-theme report (save as PDF from the print
dialog), **Download JSON** saves the full result.

For one beam it reports:

* **Flexure – learned model:** ANN (M3) residual strength % with a 90 % range, converted to kN·m
  with the IS 456 design capacity of the uncorroded section. The range combines three sources,
  each estimated from the data (see `assessment.py`):
  * *diagnostic* – the stated uncertainty of the corrosion level, pushed through the ANN;
  * *model* – held-out test error (16.6 points for corroded beams) minus specimen scatter;
  * *residual* – scatter between nominally identical specimens in the same study (13.5 points,
    from 301 replicate specimens), which no model can remove.
  The lower bound is the operative figure: on held-out studies the ANN overestimated corroded
  beams by 3.4 points on average.
* **Flexure – IS 456 area loss** for comparison.
* **Shear – code mechanics:** IS 456 cl. 40 (Tables 19, 20) with corroded tension steel and stirrups.
* **IS 456 conformity:** cover and grade for the exposure class (Tables 16, 5), w/c ratio, cover vs
  bar Ø, crack width (cl. 35.3.2), minimum/maximum tension steel after section loss, xu/d,
  minimum shear steel and stirrup spacing, and capacity vs factored demand.
* **Input completeness** and **cautions** (extrapolation, truncated upper bound, judged corrosion level, …).

The corrosion-method uncertainties (±1 / 3 / 5 / 8 points) are engineering assumptions and can be
overridden per case in the form. Stage 2 (corrosion state from NDT and visual inspection, and repair
recommendations) is not built yet.

| File | Role |
|---|---|
| `app.py` | Local web server (Python standard library only) |
| `assessment.py` | Assessment engine: inputs → flexure, shear, checks, cautions |
| `is456.py` | IS 456 tables, shear capacity and conformity checks |
| `web/` | Page, styles and script |

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `01_clean_data.py` | Reads the beam and rebar spreadsheets, converts text to numbers, removes implausible values and duplicate entries, rebuilds residual strength %, and logs every change with its Excel row number in `outputs/data/cleaning_log.csv`. |
| 2 | `02_train_models.py` | Tunes and trains five ANN models, scores them on papers never seen in training, compares against linear regression, random forest and IS 456, and saves models and figures. |
| 3 | `03_compare_analytical.py` | Residual strength by corrosion band (measured vs ANN vs IS 456), and the ABAQUS benchmark template / four-way comparison. |
| – | `predict.py` | Command-line predictions for a new beam. |
| – | `analytical.py` | IS 456 capacity formulas (same as `Parameters (input, output & condition).xlsx`, plus the Annex G over-reinforced limit). |
| – | `config.py` | Paths and assumptions (assumed fy, test fraction, ensemble size). |

## The models

| Model | Predicts | Inputs | Data |
|---|---|---|---|
| M1 | Average corrosion % of a beam | exposure days, NaCl %, half-cell potential, chloride at rebar, cover, w/c, fck, bar Ø, fibre %, pre-crack width | beams |
| M2 | Failure load (kN) | b, D, L, cover, bar Ø, no. of bars, fck, fibre %, corrosion % (+ Ast, d, ρ, L/D) | beams |
| M3 | Residual strength (% of the uncorroded control) | same as M2 | beams with a matched control |
| M4a / M4b | Rebar residual yield / ultimate stress (%) | corrosion %, bar Ø, corrosion method | rebars |

M1 → M3 is the practical chain the presentation calls for: NDT and exposure data give an
estimate of corrosion, and corrosion plus the section gives the residual strength. With the
current data the first link does not work (see Results), so a measured corrosion level is needed.

Each model is a multilayer perceptron (scikit-learn `MLPRegressor`, L-BFGS solver). Inputs are
median-imputed (with a "missing" flag), standardised, and size-type inputs are log-transformed.
The final model is the average of 5 networks trained from different random starts.

### How the models are tested

Specimens from one paper share materials, geometry and test set-up, so a random row split lets
the network "recognise" papers and overstates accuracy. Here:

* 20 % of the **source papers** are held out as the test set (`outputs/split.json`).
* Hyper-parameters (layers, activation, L2 penalty) are tuned by 5-fold **GroupKFold** on the
  training papers.
* `metrics.json` also reports the score of the same network on a random row split, to show
  how much a careless split inflates the result.

## Results (held-out papers, 15 Oct 2026)

Full tables: `outputs/model_performance.md`, `outputs/metrics.json`,
`outputs/comparison_by_corrosion_level.md`.

| Model | ANN R² | ANN mean abs. error | Best other method | IS 456 |
|---|---|---|---|---|
| M4b rebar residual ultimate stress | **0.84** | 4.9 % points | random forest 0.84 | – |
| M4a rebar residual yield stress | **0.76** | 6.1 % points | linear 0.76 | – |
| M3 beam residual strength | 0.26 | 9.7 % points | random forest 0.35 | 0.19 (RMSE 14.4 vs ANN 13.7) |
| M2 beam failure load | 0.10 | 29.8 kN | random forest 0.01 | −0.69 (conservative by design) |
| M1 beam corrosion % | 0.03 | 6.4 % | linear 0.06 | – |

What this means:

* **Rebar (M4a/b) – usable.** Residual strength of corroded bars is predicted to within about
  5–6 percentage points on studies the model never saw.
* **Beam residual strength (M3) – modest, limited by the data.** The ANN is slightly better than
  IS 456 area-loss theory, and both follow the measured trend (≈ 100 % → 86 % at 5–10 % → 74 %
  at 15–20 % corrosion). Beams at the same corrosion level differ by ±17 % between studies, which
  no model using these inputs can explain.
* **Failure load (M2) – weak.** The sheet records beam length, not clear span, and not the loading
  arrangement or steel fy; these alone change the failure load by up to 2×. Using residual % (M3)
  avoids this, because it is measured against each paper's own control beam.
* **Corrosion % (M1) – not predictable from the current inputs.** In accelerated tests corrosion is
  set by the applied current and its duration, which the beam sheet does not record.
* **A random row split would have overstated accuracy** – e.g. failure load R² 0.63 instead of 0.10 –
  because specimens from the same paper are near-duplicates.

### Next steps for the team

1. Correct the rows listed in `outputs/data/cleaning_log.csv` (especially the load and rebar
   findings below), then re-run steps 1–3.
2. Add these columns to the beam sheet: **clear span, loading type (3- or 4-point) with shear span,
   steel fy, corrosion method, and impressed current density + duration**. These are what limit M1
   and M2.
3. Model the 21 benchmark beams in ABAQUS and fill in `outputs/abaqus_benchmark_beams.csv`.

## Data-quality findings (please fix in the source spreadsheets)

1. **`Reduction in Load Capacity (%)` is broken** – in 999 of 1005 beam rows it is a copy of
   `Max Bending Moment`. It is not used. Residual strength is rebuilt as
   *load ÷ load of the uncorroded control from the same paper with identical section, span,
   reinforcement and concrete × 100*.
2. **`Max Bending Moment` is not a measurement** – it equals Load × Length / 4 in every row.
   The only measured strength value is `Load at Failure`.
3. The older `Beam dataset Year (2024-25).xlsx` is mostly free text (ranges, notes inside
   cells) and overlaps the new file in only 24 of its 371 numeric rows; it is not used.
4. The two rebar files contain identical values; `#Rebar spreadsheet.xlsx` is used.
5. **63 beam failure loads are physically impossible for the recorded section** – measured load
   is under 0.4× or over 5× the IS 456 capacity (e.g. Excel rows 696–704: 150×250 mm beams with
   2Ø10 bars at ~250 kN, 10× the capacity; rows 932–935: 6–7×). These are likely unit, dimension
   or member-type errors. They are kept for residual strength (which is relative to the same
   paper's control) but excluded from the failure-load model M2.
6. **Rebar `Engg. Nominal Ultimate Stress, Ru,eng (%)` holds another quantity in three blocks** –
   Hou et al. (2019) rows 3540–3584 (values look shifted by one column), Berrocal et al. rows
   3596–3618 and Hansapinyo et al. rows 4114–4152 show 5–18 % "residual strength" at ~0 %
   corrosion. Values below 30 % of the remaining steel fraction are blanked.
7. Everything else that was blanked or dropped is listed in `outputs/data/cleaning_log.csv`.

## Assumptions to state in the report

* Steel yield strength is not in the beam dataset; the IS 456 calculation assumes
  **fy = 500 MPa** (`config.FY_ASSUMED_MPA`).
* IS 456 uses the design formula (0.87 fy), so its absolute loads are conservative by design.
  The residual % comparison is largely free of this bias because it is a ratio.
* Pu = 4 Mu / L (central point load) with L = beam length, as in the calculation sheet.
* Corrosion ≤ 0.5 % mass loss is treated as an uncorroded control.

## ABAQUS (objectives 3 and 4)

`03_compare_analytical.py` writes `outputs/abaqus_benchmark_beams.csv`: complete beam series
(a control plus corroded beams) from the held-out papers, with measured, ANN and IS 456 results.
Model these beams in ABAQUS, enter `abaqus_load_kn` (and `abaqus_residual_pct` if you compute
it), and re-run the script to get the four-way comparison in `outputs/comparison_with_abaqus.csv`.
The script never overwrites a filled template.

## Outputs

| Path | Content |
|---|---|
| `outputs/data/beam_clean.csv`, `rebar_clean.csv` | Cleaned datasets |
| `outputs/data/cleaning_log.csv`, `cleaning_summary.txt` | What was changed and why |
| `outputs/model_performance.md`, `metrics.json` | Test-set scores of every model and baseline |
| `outputs/test_predictions/*.csv` | Per-specimen test-set predictions |
| `outputs/comparison_by_corrosion_level.md` | Measured vs ANN vs IS 456 per corrosion band |
| `outputs/figures/` | Parity plots, residual-strength curve, rebar curves, input importance |
| `models/*.joblib` | Trained models used by `predict.py` |
