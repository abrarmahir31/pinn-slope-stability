# Failure criterion re-read (scripts/recompute_criterion.py)

O-22: `three` = loss AND displacement AND admissible (the sweeps as run); `loss_floor` = loss AND admissible. Brackets come from the SRF points each sweep evaluated. **rerun?** = the loss_floor bracket is wider than 2 × bisect-tol and must be resolved by a sweep under the new criterion before it is reported.

| run | arm | w_yield | N_PDE | FOS three | = result.json | FOS loss_floor | disp fired at loss_floor failure | rerun? |
|---|---|---|---|---|---|---|---|---|
| ssr_arm_Mk_d_GSI_high | Mk_d GSI 55.0 | 1.0 | 10000 | 2.051 (2.047–2.055) | yes | 2.051 (2.047–2.055) | True | no |
| ssr_arm_Mk_d_GSI_low | Mk_d GSI 35.0 | 1.0 | 10000 | 3.215 (3.211–3.219) | yes | 3.215 (3.211–3.219) | True | no |
| ssr_arm_Mk_d_GSI_low_common | Mk_d GSI 35.0 | 1.0 | 10000 | 1.043 (1.039–1.047) | yes | 1.043 (1.039–1.047) | True | no |
| ssr_arm_Mk_d_K_s_high | Mk_d K_s x10 | 1.0 | 10000 | 1.613 (1.609–1.617) | yes | 1.613 (1.609–1.617) | True | no |
| ssr_arm_Mk_d_K_s_high_common | Mk_d K_s x10 | 1.0 | 10000 | 1.480 (1.477–1.484) | yes | 1.480 (1.477–1.484) | True | no |
| ssr_arm_Mk_d_K_s_low | Mk_d K_s x0.1 | 1.0 | 10000 | 1.496 (1.492–1.500) | yes | 1.496 (1.492–1.500) | True | no |
| ssr_arm_coupling_high | KC everywhere | 1.0 | 10000 | 1.480 (1.477–1.484) | yes | 1.480 (1.477–1.484) | True | no |
| ssr_arm_coupling_low | KC off (one-way) | 1.0 | 10000 | 1.496 (1.492–1.500) | yes | 1.496 (1.492–1.500) | True | no |
| ssr_ghb_draw11 | — | 1.0 | 10000 | 1.645 (1.641–1.648) | yes | 1.375 (1.250–1.500) | False | yes |
| ssr_ghb_draw11_lossfloor | — | 1.0 | 10000 | 1.434 (1.430–1.438) | yes | 1.434 (1.430–1.438) | True | no |
| ssr_ghb_draw12 | — | 1.0 | 10000 | 1.754 (1.750–1.758) | yes | 1.375 (1.250–1.500) | False | yes |
| ssr_ghb_draw12_lossfloor | — | 1.0 | 10000 | 1.465 (1.461–1.469) | yes | 1.465 (1.461–1.469) | True | no |
| ssr_ghb_draw13 | — | 1.0 | 10000 | 1.691 (1.688–1.695) | yes | 1.375 (1.250–1.500) | False | yes |
| ssr_ghb_draw13_lossfloor | — | 1.0 | 10000 | 1.434 (1.430–1.438) | yes | 1.434 (1.430–1.438) | True | no |
| ssr_ghb_draw14 | — | 1.0 | 10000 | 1.449 (1.445–1.453) | yes | 1.449 (1.445–1.453) | True | no |
| ssr_ghb_n20000 | — | 1.0 | 20000 | 1.621 (1.617–1.625) | yes | 1.531 (1.500–1.562) | False | yes |
| ssr_ghb_n20000_wscaled | — | 0.5 | 20000 | 1.754 (1.750–1.758) | yes | 1.375 (1.250–1.500) | False | yes |
| ssr_ghb_n20000_wscaled_lossfloor | — | 0.5 | 20000 | 1.465 (1.461–1.469) | yes | 1.465 (1.461–1.469) | True | no |
| ssr_ghb_n5000 | — | 1.0 | 5000 | 1.488 (1.484–1.492) | yes | 1.453 (1.438–1.469) | False | yes |
| ssr_ghb_n5000_wscaled | — | 2.0 | 5000 | 1.488 (1.484–1.492) | yes | 1.488 (1.484–1.492) | True | no |
| ssr_ghb_w0.3 | — | 0.3 | 10000 | 1.496 (1.492–1.500) | yes | 1.496 (1.492–1.500) | True | no |
| ssr_ghb_w1 | — | 1.0 | 10000 | 1.488 (1.484–1.492) | yes | 1.488 (1.484–1.492) | True | no |
| ssr_ghb_w1_cold | — | 1.0 | 10000 | 1.723 (1.719–1.727) | yes | 1.375 (1.250–1.500) | False | yes |
| ssr_ghb_w3 | — | 3.0 | 10000 | 1.520 (1.516–1.523) | yes | 1.520 (1.516–1.523) | True | no |
| ssr_mc_w0.3 | — | 0.3 | 10000 | 1.520 (1.516–1.523) | yes | 1.520 (1.516–1.523) | True | no |
| ssr_mc_w1 | — | 1.0 | 10000 | 1.527 (1.523–1.531) | yes | 1.527 (1.523–1.531) | True | no |
| ssr_mc_w3 | — | 3.0 | 10000 | 1.582 (1.578–1.586) | yes | 1.582 (1.578–1.586) | True | no |
