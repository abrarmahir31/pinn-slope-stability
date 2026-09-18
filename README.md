# PINN slope stability — Işıkdere open-pit lignite mine, Section 5

Physics-informed neural network for coupled hydro-mechanical slope stability,
after Ulusay et al. (2014), *Eng. Geol.* 181, 261–280. MSc thesis,
Dept. of Petroleum & Mining Engineering, SUST.

**Active branch: `step-5-strength`.** `main` lags far behind; clone with
`git clone -b step-5-strength <url>`. The branch
`claude/pinn-slope-stability-overhaul-*` is a reorganisation of the August
`main` and does **not** contain Steps 3–5; do not merge it.

## Read these first

| File | What it is |
|---|---|
| `DECISIONS.md` | **Binding.** Every modelling decision with its evidence, plus the open-issues register (O-1 …). Do not re-litigate silently. |
| `docs/RUN_ALL_STEPS.md` | Step-by-step commands from the current state to all 12 thesis figures |
| `docs/PHASE5_6_RUNBOOK.md` | Phase 5–6 decisions, evidence and reporting checklist |
| `docs/SUPERVISOR_BRIEF.md` | The Phase 5 decisions as put to the supervisor |
| `docs/open_items.md` | Older open items (Steps 1–4) |

## First run after a fresh clone

Two caches are gitignored; without them many tests fail:

    cd src/step21_geometry
    python 14_ic_checks.py       # writes ic_cache.npz
    python 17_sigma0_solve.py    # writes sigma0_cache.npz
    cd ../..
    PYTHONPATH=. python -m pytest -q     # 570 passed, 4 skipped

On Windows use `go.bat` (conda env `pinn`, `PYTHONPATH=.`).

## Layout

    src/                    library code; no script does physics of its own
      step21_geometry/      Section 5 geometry, boundary and initial conditions
      materials.py vg.py properties.py   Phase-1 parameters, VG/Mualem
      model.py loss.py mechanics.py coupling.py weighting.py   the PINN
      strength.py plasticity.py          MC and GHB criteria, SSR reduction
      failure_surface.py sensitivity.py arms.py   Step 5.3, Phase 6
    scripts/                every runnable entry point (train, sweeps, figures)
    tests/                  ~570 tests; new behaviour arrives with a test
    baselines/              frozen baselines: manifest + hashes + artefacts
    runs/                   training and sweep outputs (gitignored)
    docs/                   status notes, runbooks, figures, results

## Typical commands

    python scripts\train.py @configs\production_labpc.args --ansatz cap --seed 7 --out runs\prod_baseline_v2
    scripts\run_production_baseline.bat check       # drift, psi, loss ratios
    scripts\run_phase5.bat smoke                    # SSR smoke, both yield norms
    scripts\run_phase5.bat sweeps                   # FOS at w_yield 0.3, 1, 3
    scripts\run_phase6.bat arms                     # sensitivity arm files
    python scripts\collect_results.py --runs "runs\ssr*" --out docs\results
    python scripts\progress.py "runs\ssr*"          # progress of running sweeps

## Rules this repo runs on

1. **Verify before asserting.** Run it, read the file; don't infer from names.
2. **Untested scripts are presumed broken.** Several were found broken on first
   execution (a shell command pasted over a function body, a guard that could
   never fire, a stub shadowed by a later definition).
3. **Every claim gets a falsifiable test**, including negative controls: a test
   that cannot fail proves nothing.
4. **Only frozen baselines back headline numbers** (D-5.6). `collect_results.py`
   marks anything else `unfrozen_baseline` or `validation_only`.
5. **Decisions are recorded, not assumed.** Modelling choices have no defaults
   in the CLI; the scripts refuse to run until they are set.
