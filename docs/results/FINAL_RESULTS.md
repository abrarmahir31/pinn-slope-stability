# Final Phase 5/6 results (scripts/final_results.py)

Criterion: loss > 5 × reference AND admissible fraction (least-admissible stratum) < 0.50 — D-5.18 (displacement test dropped) and D-5.19 (physics arms use the baseline's loss reference). Baseline-v2, GHB σ₃ 0–179 kPa, raw yield norm, legacy yield reduction (D-5.17), SRF step 0.25 bisected to 0.01.

**Every FOS is calibrated (D-5.13).** The criterion was chosen to put the baseline near 1.5; the absolute values are not predictions. The relative results below are.

## 1. Collocation-draw replicates (the noise yardstick)

| run | sampling seed | FOS |
|---|---|---|
| ssr_ghb_w1 | 7 | 1.488 |
| ssr_ghb_draw11 | 11 | 1.434 |
| ssr_ghb_draw12 | 12 | 1.465 |
| ssr_ghb_draw13 | 13 | 1.434 |
| ssr_ghb_draw14 | 14 | 1.449 |

**Mean 1.454, SD 0.023 (n = 5, CV 1.6%).** The baseline seed (7) is the highest; the calibration target was met on it. Use SD = 0.023 as the resolution of a single run; the SD of a difference between two single runs is √2 × that ≈ 0.033.

## 2. w_yield plateau and MC vs GHB (Fig 10)

| w_yield | FOS_GHB | FOS_MC | MC − GHB |
|---|---|---|---|
| 0.3 | 1.496 | 1.520 | +0.023 |
| 1 | 1.488 | 1.527 | +0.039 |
| 3 | 1.520 | 1.582 | +0.062 |

Spread over w_yield 0.3–3: GHB 0.031, MC 0.062. At w_yield 1, MC − GHB = +0.039 against a difference-SD of ≈ 0.033 (1.2 SD): **not resolvable by FOS**. (Replicate SD measured for GHB only.)

## 3. Sensitivity (Fig 12) — seed-7 arms against the seed-7 baseline 1.488

| factor | low | FOS | Δ% | high | FOS | Δ% | swing |
|---|---|---|---|---|---|---|---|
| Mk_d GSI | GSI 35 | 1.043 | -29.9 | GSI 55 | 2.051 | +37.8 | 1.008 |
| Mk_d K_s | x0.1 | 1.496 | +0.5 | x10 | 1.480 | -0.5 | 0.016 |

GSI arms run against the baseline σ₀ (O-11, stated as a limitation). K_s ×10 and GSI 35 are the D-5.19 reruns.

## 4. Coupling (Fig 11)

| Kozeny–Carman feedback | FOS |
|---|---|
| off (one-way) | 1.496 |
| marls only (default) | 1.488 |
| everywhere | 1.480 |

One-way minus default: +0.008 (0.3 replicate SD).

## 5. Collocation convergence (N_PDE, matched effective w_yield, D-5.17)

| N_PDE | w_yield | FOS | change vs 10k |
|---|---|---|---|
| 5,000 | 2 | 1.488 | +0.0% |
| 10,000 | 1 | 1.488 | — |
| 20,000 | 0.5 | 1.465 | -1.6% |

Largest change from 10k: 0.023 = 1.0 replicate SD.

## 6. Every configuration: as run vs adopted criterion

| configuration | loss reference | FOS as run | FOS adopted | bracket | how |
|---|---|---|---|---|---|
| ssr_arm_Mk_d_GSI_high | baseline (D-5.19) | 2.051 | **2.051** | 2.047–2.055 | as-run sweep re-read (bracket resolved) |
| ssr_arm_Mk_d_GSI_low | baseline (D-5.19) | 3.215 | **1.043** | 1.039–1.047 | rerun under the adopted criterion |
| ssr_arm_Mk_d_K_s_high | baseline (D-5.19) | 1.613 | **1.480** | 1.477–1.484 | rerun under the adopted criterion |
| ssr_arm_Mk_d_K_s_low | baseline (D-5.19) | 1.496 | **1.496** | 1.492–1.500 | as-run sweep re-read (bracket resolved) |
| ssr_arm_coupling_high | baseline (D-5.19) | 1.480 | **1.480** | 1.477–1.484 | as-run sweep re-read (bracket resolved) |
| ssr_arm_coupling_low | baseline (D-5.19) | 1.496 | **1.496** | 1.492–1.500 | as-run sweep re-read (bracket resolved) |
| ssr_ghb_draw11 | own | 1.645 | **1.434** | 1.430–1.438 | rerun under the adopted criterion |
| ssr_ghb_draw12 | own | 1.754 | **1.465** | 1.461–1.469 | rerun under the adopted criterion |
| ssr_ghb_draw13 | own | 1.691 | **1.434** | 1.430–1.438 | rerun under the adopted criterion |
| ssr_ghb_draw14 | own | 1.449 | **1.449** | 1.445–1.453 | as-run sweep re-read (bracket resolved) |
| ssr_ghb_n20000 *(evidence only)* | own | 1.621 | **—** | 1.500–1.562 | UNRESOLVED: rerun needed |
| ssr_ghb_n20000_wscaled | own | 1.754 | **1.465** | 1.461–1.469 | rerun under the adopted criterion |
| ssr_ghb_n5000 *(evidence only)* | own | 1.488 | **—** | 1.438–1.469 | UNRESOLVED: rerun needed |
| ssr_ghb_n5000_wscaled | own | 1.488 | **1.488** | 1.484–1.492 | as-run sweep re-read (bracket resolved) |
| ssr_ghb_w0.3 | own | 1.496 | **1.496** | 1.492–1.500 | as-run sweep re-read (bracket resolved) |
| ssr_ghb_w1 | own | 1.488 | **1.488** | 1.484–1.492 | as-run sweep re-read (bracket resolved) |
| ssr_ghb_w1_cold | own | 1.723 | **—** | 1.250–1.500 | UNRESOLVED: rerun needed |
| ssr_ghb_w3 | own | 1.520 | **1.520** | 1.516–1.523 | as-run sweep re-read (bracket resolved) |
| ssr_mc_w0.3 | own | 1.520 | **1.520** | 1.516–1.523 | as-run sweep re-read (bracket resolved) |
| ssr_mc_w1 | own | 1.527 | **1.527** | 1.523–1.531 | as-run sweep re-read (bracket resolved) |
| ssr_mc_w3 | own | 1.582 | **1.582** | 1.578–1.586 | as-run sweep re-read (bracket resolved) |

The cold-start check under the adopted criterion is bracketed only to 0.25 (it contains the warm value); the 5,000-epoch cold step of 22 Sep remains the evidence that warm-starting is unbiased.
