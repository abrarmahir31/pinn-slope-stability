# Phase 5–6 runbook (Day 40)

What exists after Day 40, what you must decide before GPU time, how to run it,
and what must be reported next to every number. Nothing here is a result.
Every FOS, figure and table comes from your runs.

---

## 1. Decisions — none has a default in the code

| # | Decision | Where it is set | Evidence to bring to the supervisor |
|---|---|---|---|
| D1 | Soft constraint (current) vs return mapping | method choice; `ssr_sweep.py` is the soft arm | `STEP5_STATUS.md` §"What the sweep would need"; FOS depends on `w_yield` |
| D2 | MC strength for SSR (see §2.1) | `ssr_sweep.MC_DESIGN`, `RUN_MC` in `run_phase5.bat` | §2.1 numbers |
| D3 | GHB σ₃ fit range | `SIG3_LO`, `SIG3_HI` | `scripts/sig3_range.py`, §2.3 |
| D4 | D = 1 for all strata | `properties.py` / Phase-1 dataset | Task 1: Mk_d rock-mass UCS 41 kPa; 55% of Mk_d outside GHB at SRF = 1 |
| D5 | Admissible metric and drop | `ADM_METRIC` (`area`, `min_tag`, `tag:Mk_d`), `ADM_DROP` | §2.4 |
| D6 | Yield-term normalisation | `YIELD_NORM` (`raw` or `ref`) | smoke: L_yield(SRF 1) = 69.3 MC vs 0.026 GHB |
| D7 | `w_yield` reported in the thesis | `W_REPORT` (Phase 6) | Fig 10 plateau |
| D8 | Critical state for Fig 9 | `FIG9_STATE` (`failed` or `stable`) | caption |
| D9 | K_s arm stratum (replaces coal seam) | `KS_STRATUM` | Section 5 has no coal |
| D10 | GSI arm stratum | `GSI_STRATUM` | §2.6 |
| D11 | Rainfall arm: run or drop | `make_arms.py --include-rainfall` | §2.2 |
| D12 | Network-seed replicates: retrain or collocation draws only | §3 | cost |
| D13 | Tension clip in `make_fig6` FS | `hoek_brown_sigma1` | 147 points hidden by the clip |
| D14 | `--plateau-factor` 20, `--disp-factor` 10 | CLI, written to result.json | inherited Day-39 values, never calibrated |

---

## 2. Day 40 findings that bear on those decisions

### 2.1 The MC sweep as specified starts almost fully yielded

`MC_DESIGN` is c = 4.4 kPa, φ = 15.4°: Ulusay's smooth bedding-plane
**residual**, a discontinuity strength. `ssr_sweep.py` applies it as a
continuum criterion to every point of every stratum, Tm included.
`plasticity.reduced_material_params` warns about exactly this.

Smoke run on `baseline-v1` (CPU, N_PDE = 400, 50 epochs; indicative only):

| | L_yield at SRF 1, baseline net | admissible after SRF 1 fine-tune (Mk / Mk_d / Tm) |
|---|---|---|
| MC | 69.3 | 0.03 / 0.09 / 0.42 |
| GHB | 0.026 | 1.00 / 0.97 / 1.00 |

Under MC admissibility **rose** with SRF (0.38 → 0.60 area metric) as the
penalty reshaped the field, so the drop condition cannot fire. An MC sweep
run as-is would spend GPU hours to report "no failure", or an artefact.
Options for D2 include the Phase-1 rock-mass MC equivalents, restricting the
bedding pair to a bedding layer, or dropping MC-SSR. Not chosen here.

### 2.2 The rainfall arm cannot move FOS

Rain enters only on `natural_ground` and `bench`, which are 100% Mk_d at
K_s = 1e-9 m/s. The capacity cap binds at every point, so 10, 20 and 40 mm/hr
are **bit-identical** boundary conditions. The cut face is no-flow.
`tests/test_arms.py::test_rainfall_arms_are_currently_identical_boundary_conditions`
pins this and fails the day it changes. Separately, `flux_bc`'s docstring says
about 56% infiltrates "on Tm", but no rainfall segment touches Tm.

### 2.3 σ₃ in the baseline (`scripts/sig3_range.py`, t = 30 d, area-weighted)

| zone | p5 | p50 | p95 | p99 | tension | ≤ 850 kPa |
|---|---|---|---|---|---|---|
| Mk | −5.8 | 33.5 | 98.4 | 133.2 | 10.1% | 100% |
| Mk_d | −14.1 | 50.6 | 130.7 | 179.4 | 12.1% | 100% |
| Tm | 237.3 | 561.1 | 1072.2 | 1247.6 | 0% | 82.2% |
| all, pooled | 31.3 | 491.7 | 1057.3 | 1240.0 | 1.5% | 84.5% |

kPa. The Day-39 default of 850 kPa sits near **Tm's** p75, about 6× beyond
the marls' p95. Rerun the script with `--tags` and `--max-depth-m` for the
zone you choose, and keep its JSON.

### 2.4 Why the admissible test is a drop, not a floor

Task 1: 55% of Mk_d (5.4% of the area) is outside GHB in the elastic
baseline. The Day-39 absolute floor of 0.80 would read `tag:Mk_d` as failed at
SRF = 1. With `area`, Tm holds 87.2% of the area at 100% admissible, so even
complete yielding of Mk and Mk_d moves the metric by at most **0.068**
(`area` 0.940 → 0.872). `--admissible-metric area --admissible-drop 0.10`
therefore cannot fire unless Tm yields. `ssr_sweep.py` now warns about this at
the reference state (`reachable_drop` in result.json). Choose the metric and
drop together; see `docs/SUPERVISOR_BRIEF.md` D5.

### 2.5 Other items from Days 40–41

- The fixed base (u = v = 0) moves ~1.3–1.4 mm by t = 30 d on `baseline-v1`.
- `baseline-v1` is 2,000 epochs from a dirty tree, not the 100k production run.
- The yield check is on total stress including the Bishop increment
  (`loss.L_yield` docstring); `plasticity.yield_violation` describes its
  inputs as effective stress. Reconcile in Methodology.
- `STEP6_STATUS.md`'s ~50 GPU-days for the N_PDE ladder scaled from
  `train.py`'s default N = 500. Production is 10,000; the ladder is a few
  GPU-days.
- Strike "~21.5%" from the plan (STEP6_STATUS §2). Report whatever
  `coupling_effect` returns, with its sign.

### 2.6 The GSI arm and σ₀

The GSI arm moves E with m_b, s, a (correctly, per `sensitivity.py`). σ₀ comes
from the offline FE gravity solve with baseline moduli and is **not**
re-equilibrated. `result.json` records `sigma0_consistent: false` for those
arms. Re-solving σ₀ per arm (`src/step21_geometry/17_sigma0_solve.py`) is a
decision.

---

## 3. Order of operations

Wall-clock figures are estimates. Read `sec` per SRF from the smoke log and
rescale. A sweep is roughly 8–13 SRF evaluations × 1,500 epochs.

| step | command | notes |
|---|---|---|
| 0 | `git push origin step-5-strength` | Day 40 commits are local only |
| 1 | `python -m pytest -q` | expect the count in the commit message |
| 2 | `python scripts\sig3_range.py <ckpt> --tags Mk_d` | D3 input |
| 3 | fill DECISIONS in `scripts\run_phase5.bat` | aborts until done |
| 4 | `scripts\run_phase5.bat smoke` | first GPU execution; read per-stratum `adm` |
| 5 | `scripts\run_phase5.bat sweeps` | 3 GHB (+3 MC if D2 allows) |
| 6 | `scripts\run_phase5.bat cold` | cold-start check at w = 1 |
| 7 | `scripts\run_phase5.bat figs` | `docs\results\fos_table.md`, Figs 8, 9, 10 |
| 8 | choose `W_REPORT` from Fig 10; fill `run_phase6.bat` | D7, D9, D10 |
| 9 | `scripts\run_phase6.bat arms` then `sweeps` | K_s, GSI, coupling arms |
| 10 | `scripts\run_phase6.bat npde` | 5k and 20k; 10k is the step 5 run |
| 11 | `scripts\run_phase6.bat draws` | collocation-draw spread |
| 12 | `scripts\run_phase6.bat figs` | Figs 11, 12, updated table |

**Network-seed replicates (D12).** `--sampling-seed` varies the collocation
draw, not network initialisation. True seed replicates need one baseline per
seed, e.g.
`python scripts\train.py @configs\production_labpc.args --seed 11 --out runs\prod_seed11`,
then a sweep per baseline. At 100k epochs that is ~16 h per seed before its
sweep.

---

## 4. Reporting checklist — attach to every FOS

- [ ] criterion, and for MC which strength pair (D2)
- [ ] `w_yield` and the plateau across 0.3 / 1 / 3 (Fig 10 spread)
- [ ] `yield_norm`, `admissible_metric`, `admissible_drop`, plateau and disp factors
- [ ] σ₃ fit range and the zone and percentiles it came from (GHB)
- [ ] bracket and bisection tolerance
- [ ] warm vs cold-start difference
- [ ] KC arm, N_PDE, baseline checkpoint and its epoch count
- [ ] for Fig 9: SRF, `FIG9_STATE`, LI and LI/LI(start), band width at two grids
- [ ] `sigma0_consistent` for any GSI arm
- [ ] `git_commit` from result.json
- [ ] comparison with LEM F = 0.94 stated as residual-bedding LEM vs PINN criterion used

---

## 5. Results → thesis map (structure only)

| Thesis item | Source | Generated by |
|---|---|---|
| Table: FOS by criterion and w_yield | `docs/results/fos_table.md` | `collect_results.py` |
| Fig 6 + masked recount text | `docs/figs/fig6*.json` | `make_fig6.py` |
| Fig 8 loss / admissible vs SRF | sweeps | `make_ssr_figs.py` |
| Fig 9 γ_max and surfaces | `states/` | `make_fig9.py` |
| Fig 10 FOS vs w_yield | plateau groups | `make_ssr_figs.py` |
| Fig 11 coupling | coupling arms | `make_ssr_figs.py` |
| Fig 12 tornado | OAT arms | `make_ssr_figs.py` |
| Methods: σ₃ range | `docs/results/sig3_range.json` | `sig3_range.py` |
| Limitations | §2.1, §2.2, §2.5, §2.6 | this file |

---

## 6. Files added on Day 40

`scripts/ssr_sweep.py` (rewritten), `src/arms.py`, `scripts/sig3_range.py`,
`scripts/make_arms.py`, `scripts/collect_results.py`,
`scripts/make_ssr_figs.py`, `scripts/make_fig9.py`,
`scripts/run_phase5.bat`, `scripts/run_phase6.bat`; tests
`test_ssr_sweep.py`, `test_arms.py`, `test_sig3_range.py`,
`test_collect_results.py`, `test_ssr_figs.py`; `.gitattributes` adds
`*.bat text eol=crlf`.

Not done, because it can't be done in advance: every FOS, Figs 8–12, choosing
among the options in §1, and the thesis prose.
