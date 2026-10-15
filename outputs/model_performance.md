# Test-set performance (held-out papers)

| Model | Train / test rows (papers) | Method | R² | RMSE | MAE |
|---|---|---|---|---|---|
| M1_beam_corrosion | 737 / 219 (128 / 33) | ANN | 0.03 | 11.11 % | 6.43 % |
| M1_beam_corrosion |  | Linear (ridge) | 0.06 | 10.92 % | 6.69 % |
| M1_beam_corrosion |  | Random forest | 0.01 | 11.22 % | 6.17 % |
| M2_beam_failure_load | 686 / 191 (104 / 27) | ANN | 0.10 | 37.70 kN | 29.82 kN |
| M2_beam_failure_load |  | Linear (ridge) | -1.12 | 58.06 kN | 40.91 kN |
| M2_beam_failure_load |  | Random forest | 0.01 | 39.64 kN | 31.50 kN |
| M3_beam_residual_strength | 581 / 179 (86 / 24) | ANN | 0.26 | 13.74 % | 9.67 % |
| M3_beam_residual_strength |  | Linear (ridge) | 0.28 | 13.59 % | 9.53 % |
| M3_beam_residual_strength |  | Random forest | 0.35 | 12.87 % | 9.01 % |
| M4a_rebar_residual_yield | 2430 / 354 (497 / 128) | ANN | 0.76 | 8.32 % | 6.12 % |
| M4a_rebar_residual_yield |  | Linear (ridge) | 0.76 | 8.21 % | 6.31 % |
| M4a_rebar_residual_yield |  | Random forest | 0.74 | 8.54 % | 5.67 % |
| M4b_rebar_residual_ultimate | 2441 / 393 (519 / 131) | ANN | 0.84 | 6.82 % | 4.89 % |
| M4b_rebar_residual_ultimate |  | Linear (ridge) | 0.82 | 7.25 % | 5.44 % |
| M4b_rebar_residual_ultimate |  | Random forest | 0.84 | 6.89 % | 4.60 % |
