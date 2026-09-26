# Overnight run summary (scripts/run_remaining.py)

Started 2026-09-25 03:12, finished 2026-09-26 04:53. Baseline-v2 (`3345c50c…`), calibrated criterion D-5.13, w_yield 1 (N_PDE runs scaled per D-5.17).

pytest: passed

| run | group | status | FOS | bracket | hours | note |
|---|---|---|---|---|---|---|
| ssr_ghb_n5000_wscaled | npde | done earlier | 1.488 | [1.484375, 1.492188] | — |  |
| ssr_ghb_n20000_wscaled | npde | done earlier | 1.754 | [1.75, 1.757812] | — |  |
| ssr_arm_coupling_low | coupling | fos | 1.496 | [1.492188, 1.5] | 1.86 |  |
| ssr_arm_coupling_high | coupling | fos | 1.480 | [1.476562, 1.484375] | 2.33 |  |
| ssr_arm_Mk_d_K_s_low | ks | fos | 1.496 | [1.492188, 1.5] | 2.16 |  |
| ssr_arm_Mk_d_K_s_high | ks | fos | 1.613 | [1.609375, 1.617188] | 2.44 |  |
| ssr_arm_Mk_d_GSI_low | gsi | fos | 3.215 | [3.210938, 3.21875] | 4.05 |  |
| ssr_arm_Mk_d_GSI_high | gsi | fos | 2.051 | [2.046875, 2.054688] | 2.97 |  |
| ssr_ghb_draw11 | draws | fos | 1.645 | [1.640625, 1.648438] | 2.42 |  |
| ssr_ghb_draw12 | draws | fos | 1.754 | [1.75, 1.757812] | 2.7 |  |
| ssr_ghb_draw13 | draws | fos | 1.691 | [1.6875, 1.695312] | 2.43 |  |
| ssr_ghb_draw14 | draws | fos | 1.449 | [1.445312, 1.453125] | 2.15 |  |

Tables: `fos_table.md` (plateau, tornado, coupling, replicates, N_PDE at matched effective w_yield) and `criterion_o22.md` (every run under both failure criteria).

GSI arms ran against the baseline sigma_0 (O-11): state it as a limitation, or re-solve sigma_0 and rerun those two arms.
