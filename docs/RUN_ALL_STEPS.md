# Run everything: from the probes to all 12 figures (Day 42)

Every figure and table in Results comes from one of the commands below, on
the lab PC, from `C:\work\pinn-slope-stability` after `go.bat`. Each step says
**what genuine output looks like** and **what to do if it doesn't**. Open
decisions that block a step are marked ⛔ and listed in `DECISIONS.md`
("Open issues register").

Rules that keep the data authentic:
- Never type a number into a figure or table. Tables come from `result.json`
  via `collect_results.py`; figures come from checkpoints and logs.
- Only runs marked `headline = eligible` in `docs\results\fos_table.md` go in
  the thesis (D-5.6). Smoke runs and `runs\probe_*` are never collected.
- Commit after every step that writes to `docs\`, so each figure has a commit.
- Keep one window per long run and don't close it. If a run dies, rerun the
  same command: sweeps resume from `states\`.

Check progress from a second window at any time:

```
python scripts\progress.py "runs\probe_*" "runs\ssr*"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv -l 10
```

---

## Step 1 — Finish the probes and settle the detector (tonight)

`probe_ghb_ref` is done; `probe_ghb_raw` is running.

```
python scripts\progress.py runs\probe_ghb_ref runs\probe_ghb_raw
```

**Genuine output:** five SRF lines per run (1.00–2.00), ~16 min each, Mk and
Tm near 1.000, Mk_d falling.

**Then decide** ⛔ O-1 `YIELD_NORM` and ⛔ O-2 `ADM_DROP` / detector. Record in
`DECISIONS.md`, then set both in `scripts\run_phase5.bat` and
`scripts\run_phase6.bat`.

---

## Step 2 — Production baseline (~18 h GPU) → Figs 2, 3, 4, 5, 6

⛔ O-3: set `SEED` in `scripts\run_production_baseline.bat` first.

```
scripts\run_production_baseline.bat train
```
**Genuine output:** `runs\prod_baseline_v2\log.jsonl` grows every 100 steps;
~0.57 s/epoch; ends with an L-BFGS phase of 500 iterations.
**If** it refuses because the log exists: a previous attempt is there. Resume
by hand or choose a new `OUT`.

```
scripts\run_production_baseline.bat check
```
**Genuine output:** a two-column table, `prod_baseline_v2` vs `exp_seed7`,
with base drift in mm, `psi_max`, and `L/L0` per term; `FACTS: none failing`.
Compare: baseline-v1 has base drift 1.29 mm (u) / 1.40 mm (v) at t = 30 d and
`bc_mech` L/L0 = 1.3e-2. **If** `pde_richards` L/L0 is ~0.4 with `psi_max` pinned
near 0, that is the D-A.1 cap failure: stop and report, do not freeze.
⛔ O-6: whether the drift is acceptable is a decision; write it down.

```
scripts\run_production_baseline.bat fig6
scripts\run_production_baseline.bat freeze
python scripts\freeze_baseline.py --verify baselines\baseline-v2
```
**Genuine output:** freeze prints `accepted under D-5.4`; verify reports every
hash OK. **If** freeze refuses for a dirty tree: commit first, never `--force`.

```
scripts\run_production_baseline.bat figs
git add docs\figs docs\results baselines\baseline-v2
git commit -m "Production baseline v2: check, freeze, Figs 2-6"
```
**Genuine output:** `docs\figs\fig2.png` … `fig6.png` plus `fig3.json`,
`fig6.json`. Fig 2's optimiser box must read **Adam 100,000 → L-BFGS 500**;
if it reads 2,000 it was drawn from baseline-v1.

---

## Step 3 — Phase 5 on the production baseline (~20–26 h) → Figs 8, 9, 10, S1

In `scripts\run_phase5.bat` set
`BASELINE=runs\prod_baseline_v2\ckpt_final.pt`. ⛔ O-4 `FIG9_STATE`,
⛔ O-5 Fig 10 layout.

```
scripts\run_phase5.bat sweeps
scripts\run_phase5.bat cold
scripts\run_phase5.bat figs
```
**Genuine output:** `runs\ssr_ghb_w0.3`, `_w1`, `_w3`, `runs\ssr_mc_w*`,
`runs\ssr_ghb_w1_cold`, each with `result.json`. `collect_results` prints
the run count and `plateau: 2 group(s)`. `docs\results\fos_table.md` shows
`headline = eligible` on every row. Fig 9 prints `LI/LI(start)` per run.
**If** any row says `validation_only:baseline-v1`, `BASELINE` still points at
v1. **If** a sweep says `NO FAILURE up to SRF 3.000`, report it as a finding;
do not raise `--srf-max` or lower thresholds to force a FOS.
**If** Fig 9's `LI/LI(start)` is near 1, the red line is not a slip surface.

```
git add docs\figs docs\results
git commit -m "Phase 5 production: FOS table, Figs 8 9 10 S1"
```

---

## Step 4 — Phase 6 (~40 h) → Figs 11, 12

In `scripts\run_phase6.bat` set `BASELINE` to the production checkpoint and
⛔ O-9 `KS_STRATUM`, `GSI_STRATUM`.

```
scripts\run_phase6.bat arms
scripts\run_phase6.bat sweeps
scripts\run_phase6.bat npde
scripts\run_phase6.bat draws
scripts\run_phase6.bat figs
git add docs\figs docs\results configs\arms
git commit -m "Phase 6: sensitivity, coupling, N_PDE, draws, Figs 11 12"
```
**Genuine output:** `make_arms` lists 6 files, 2 marked
`[moves E: sigma_0 not re-equilibrated]`. `collect_results` reports
`tornado`, `coupling`, `n_pde`, `replicates` groups.
**If** the coupling bar is ~0%: that is consistent with D-A.4 (rain enters
at 1e-9 m/s) and is a result, not a failure.

---

## Step 5 — Figs 1 and 7

- **Fig 7** is final and baseline-independent. Regenerate only to change
  styling: `python scripts\verify_nguyen_raudkivi.py --out docs\figs\fig7.png`
- **Fig 1** (site map and section) cannot be generated from the repo. Redraw
  from Ulusay et al. (2014) with a citation; reproducing their figure needs
  publisher permission.

---

## Step 6 — Authenticity check before writing numbers

```
python scripts\collect_results.py --runs "runs\ssr*" --out docs\results
python scripts\freeze_baseline.py --verify baselines\baseline-v2
python -m pytest -q
git status --short
```
Check, for every number you will quote:
- the row in `fos_table.md` is `headline = eligible`;
- its `result.json` `git_commit` is a commit on `step-5-strength`;
- its `baseline_sha256` equals the checkpoint hash in
  `baselines\baseline-v2\MANIFEST.json`;
- `git status` is clean, so the tables match committed code.

## Figure map

| Fig | Command (step) | Source |
|---|---|---|
| 1 | manual redraw | Ulusay et al. 2014 |
| 2 | `run_production_baseline.bat figs` (2) | production ckpt cfg |
| 3 | `run_production_baseline.bat figs` (2) | geometry + sampler |
| 4 | `run_production_baseline.bat figs` (2) | production log |
| 5 | `run_production_baseline.bat figs` (2) | production ckpt |
| 6 | `run_production_baseline.bat figs` (2) | production ckpt |
| 7 | final (5) | verification run |
| 8 | `run_phase5.bat figs` (3) | sweeps |
| 9 | `run_phase5.bat figs` (3) | sweep states |
| 10 | `run_phase5.bat figs` (3) | FOS table |
| S1 | `run_phase5.bat figs` (3) | plateau |
| 11 | `run_phase6.bat figs` (4) | coupling arms |
| 12 | `run_phase6.bat figs` (4) | OAT arms |
