# Phase 1 — Consolidated Reference State

**Işıkdere lignite open-pit mine · PINN coupled hydro-mechanical slope stability**
Primary source: Ulusay, Ekmekçi, Tuncay & Hasançebi (2014), *Engineering Geology* 181, 261–280.

This folder replaces every Phase-1 data file you had. Import one module, get everything.

```python
from isikdere_phase1 import DATA, STRATA, SCALES, vg_K, vg_C, kozeny_carman

marl = STRATA["marl_sekkoy"]
K = vg_K(psi, marl)                       # works on floats, numpy arrays, torch tensors
n = marl.n0 + eps_v                       # porosity update
K_updated = kozeny_carman(n, marl)        # two-way coupling, no outer iteration
```

## Files

| File | What it is |
|---|---|
| `isikdere_phase1_dataset.json` | **The single data file.** Strata, Hoek–Brown, MC strength, geometry, BC/IC, validation targets, non-dimensional scales, conflict register. Strict SI. |
| `isikdere_phase1.py` | Loader + constitutive functions (van Genuchten–Mualem, Kozeny–Carman, Hoek–Brown, non-dimensional residuals). Import this, not the JSON directly. |
| `test_isikdere_phase1.py` | Verification suite. Run it after any edit — it catches unit slips, broken derivatives and missing parameters. Currently all checks pass. |
| `isikdere_phase1_strata.csv` | Flat strata table for eyeballing in Excel. Convenience only; the JSON is authoritative. |

**Superseded — do not import from these any more:** `phase1_reference_state.json/.csv`, `step_1_2_1_3_adopted_parameters.json/.csv`, `nondim.py`. Each contains at least one value corrected below.

## Ten source conflicts, and what was done about them

Your sources disagreed with each other in ten places. A straight merge would have carried the errors into the simulation. Each is recorded in `DATA["conflicts_resolved"]`; run `print_conflicts()` to read them in full.

**Three matter enough to state plainly:**

**C2 — the seepage-face flux was wrong by a factor of ten.** The paper's text prints q₁ = 0.056 m³/day/m, and that value propagated into your Step 1.1 SI sheet as 6.48 × 10⁻⁷ m³/s/m. The plotted Fig. 9 curve starts at 0.56. Summing 0.56/√t over 30 days gives 5.368 m³/m, matching the paper's *own* published cumulative of 5.37 to three significant figures; the 0.056 version gives 0.537. The text carries a typo. **Adopted 6.481 × 10⁻⁶ m³/s/m.** This is a boundary condition — left uncorrected it would have invalidated every transient result you produced.

**C1 — marl stiffness was roughly double the tabulated value.** The roadmap used MR = 400 for the marl, giving E_rm ≈ 478 MPa. Hoek & Diederichs (2006) tabulate MR = 150–200 for marls. **Adopted MR = 175 → E_rm = 208.9 MPa**, with 478 MPa retained as the upper sensitivity case.

**C10 — coal-seam K and S are mutually inconsistent, and this one is *not* resolved.** The packer tests give K = 0.1–0.5 m/day. The Fig. 10a drawdown curve fits w = T/S = 1850 m²/day, which with the paper's S = 3 × 10⁻⁵ implies T = 0.056 m²/day — two orders below the packer K for any plausible seam thickness. That is a genuine ambiguity in the source, not a digitisation artefact. Both values ship in the dataset. **You must pick one, justify it in the methodology, and show the other as a sensitivity case.** The code will not choose for you.

The remainder: n₀ recomputed from each stratum's own unit weight (the weak-zone marl had inherited the marl's porosity — 0.427 where the phase relation gives 0.350, and that stratum is the one that actually fails); the Step 1.2 Rosetta3 SWCC set adopted over the roadmap placeholders; ν = 0.35 for the underclay; E_ref lowered from 1 GPa to 100 MPa (no stratum in your model is as stiff as 1 GPa); and three published-figure errors corrected (Fig. 8 axis assignment, Fig. 11 normal-stress mapping, Fig. 19e's F = 1.250 vs the text's 1.3).

## Two things the dataset will not let you forget

**Π_R,diff = 3.0 × 10⁻⁶ is not a bug.** With T_ref = 1 day the Richards diffusion group sits six orders below unity because the pit's diffusive timescale is ~916 years — a pressure signal genuinely barely moves in a day. Non-dimensionalising the *variables* does not by itself put every residual *term* at O(1). Keep T_ref = 1 day (the rainfall transient is your phenomenon), normalise the seepage residual explicitly in the loss, and report Π_R,diff so the weighting is reproducible. Reviewers of PINN papers look for exactly this.

**Section geometry is Grade C.** The Fig. 19 profiles are digitised from small rasters with vertical exaggeration: ±10 m. A 10 m error in slope height moves F by more than the 5–10 % agreement band your proposal commits to. The coordinates are adequate to size the collocation domain and place boundary conditions while you build the code — not to produce a publishable SSR number. Re-digitise from the vector PDF before final runs.

## Blocking item for Phase 2

Rainfall intensity time series, MGM Muğla station. INS-2 and INS-3 followed a heavy-rainfall week; the surface Neumann flux is the forcing the entire transient case rests on. Everything else in Phase 1 is closed.

## Phase-1 exit checklist

For all six strata: K_s ✔ ρ_dry ✔ ρ_sat ✔ E ✔ ν ✔ (α, n, θ_s, θ_r, m) ✔ n₀ ✔
For the four rock strata: (σ_ci, m_i, GSI, D, m_b, s, a) ✔
Coupling closed: porosity update → Kozeny–Carman → effective stress ✔
Non-dimensional scales and groups ✔ · 28 LEM benchmarks ✔ · 23 monitoring wells ✔ · 30-day inflow series ✔

Verified by `test_isikdere_phase1.py`, which re-derives n₀ from raw unit weights, checks the analytic moisture capacity against a numerical derivative of θ(ψ), reproduces the Step 1.5 Hoek–Brown and Mohr–Coulomb tables, confirms the corrected flux reproduces the published 30-day mass balance, and round-trips the non-dimensionalisation to 10⁻¹³.
