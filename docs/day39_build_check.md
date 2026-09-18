# Build check, bug fixes, and the compute reality

Run against `pinn-full.zip` / `pinn-slope-stability-step5.zip` on a 1-core,
3 GB, no-GPU container. torch 2.14.0, float64, second-order autograd verified.

---

## 1. Build state

`pytest -q` -> **445 passed, 4 skipped, 1 warning, ~23 s.** No regression
against the Day 38 count.

The one warning is the deliberate negative control in
`test_ghb_tensile_clamp_returns_finite_below_the_cutoff`, which calls
`ghb_sigma1(..., clamp_tensile=False)` so that numpy raises on the fractional
power of a negative base. That is the failure the clamp exists to prevent, so
the warning is the test working.

**The two archives are the same code.** Every file that `diff -rq` reported as
different between `pinn-full` and `pinn-slope-stability-step5` was identical
after stripping `\r`. Working tree == committed `step-5-strength`. Nothing
uncommitted is at risk, and the CRLF/LF split is the expected
`core.autocrlf` behaviour, not drift.

---

## 2. Bugs found

All three are in `scripts/`. The test suite covers `src/` only, which is why
none of them were caught.

### B-1 `scripts/derive_step_1_2_1_3.py` -- SyntaxError, file will not parse

The body of `run_rosetta()` had been overwritten by a pasted shell command:

```python
def run_rosetta(rows, version=3):
    """rows: [sand, silt, clay, (bd)]; ..."""
    sed -n '1,10p;30,40p' derive_step_1_2_1_3.py   # <-- not Python
    return mean, stdev, codes
```

**Why it matters more than the other two.** This script is the provenance of
the Step 1.2/1.3 SWCC and elastic parameters, which are in the thesis tables.
Until now those numbers had no runnable derivation behind them.

Two further defects in the same file:

- `SoilData` was never imported, so the restored body would have failed anyway.
- Output path `../data/derived_raw.json` assumes cwd == `scripts/`, which
  contradicts the run-from-root convention the rest of the repo uses. Now
  routed through `scripts/paths.py`.

**Verification.** Re-running the restored script regenerates
`data/derived_raw.json` with an identical key set (186 leaves) and a maximum
relative difference of **2.1e-15** against the committed original -- pure
float round-off. The reconstruction is the original function, not a plausible
substitute. The `rosetta-soil` 0.1.2 signature
`rosetta(version, SoilData.from_array(rows)) -> (mean, stdev, codes)` matches
the surviving docstring's column list exactly, so there was only one thing the
body could have been.

### B-2 `scripts/check_kc_range.py` -- crashes on its final line

The script probes `tag="XX"` to document that `KCConfig.is_on` "fails open" on
unknown stratum tags. `is_on` was later hardened to raise `KeyError`. The
script was never updated, so it dies after printing all of its real output,
and its closing line asserts the opposite of current behaviour.

This matters because this script is the standing evidence that Kozeny-Carman
feedback is live and that Tm's `per_unit` exclusion is deliberate (D-3.3.2).
It should exit 0.

Now wrapped in `try/except KeyError`, reporting **fails CLOSED, loudly**, with
an `AssertionError` on the `else` branch so that a future regression back to
silent fall-through fails the script rather than passing quietly.

Output unchanged and still matches the Day 24 record: Mk_d 1.1225 at
eps_v = 1e-2, Tm exactly 1.0.

### B-3 `scripts/check_float32.py` -- a guard that could never fail

`residual_for()` ended with `out[tag] = R.detach().double()`, casting back
*before* storing. The line below printed `b.dtype` with the comment
`# sanity: must be from float32 path` -- which read `float64` unconditionally.

If the float32 arm had silently run in float64, the script would have compared
a number with itself and reported "float32 is fine" for every stratum. That is
precisely the failure mode the check exists to detect.

**The conclusion still stands.** Relative errors of ~2e-7 are float32 epsilon,
so the arithmetic genuinely ran in float32 and float64 remains justified
(`derivatives.py:56`). It was the guard that was broken, not the result.

Now stores `(values, dtype)` and asserts both arms, per the negative-control
rule in `step34_handoff.md`.

### Not a bug: `src.config.BOUNDS`

The Day 26 open item recorded this as stale (`x_max 4.41, z_max 1.18`). It is
not, any more -- it now agrees with `bounds_from_geometry(t_max=30, s=SCALES)`
to 3-4 decimals. That open item can be closed.

It remains a *rounded duplicate* with a live `TODO` at `config.py:20`, giving a
~0.03% discrepancy in `x_max` and two sources of truth for the net's input
scaling. Worth collapsing into a call, but it is tidy-up, not correctness.

---

## 3. The full run at 100k epochs -- why it did not happen

Two independent blockers.

### 3a. The yield term is still unwired -- this is the real one

> **CORRECTED (Day 42).** Superseded later the same day: `92a8683`
> "feat(loss): wire pde_yield into total_loss (soft constraint, Step 5.1)".
> `total_loss` now takes `yield_params`, `w_yield` and `yield_criterion` and
> reports `pde_yield` plus per-tag `admissible_*`. The finding below was true
> when written; it is kept as the record of why the term was wired.

`src/loss.py` `total_loss` has **no** `yield_params` argument and **no**
`pde_yield` entry in its parts dict. Confirmed by grep; unchanged since
`STEP5_STATUS.md` was written.

`scripts/ssr_sweep.py:180` calls
`total_loss(..., yield_params=yparams, w_yield=a.w_yield)` and will raise
`TypeError` on first contact. It has still never executed.

So a 100,000-epoch run today would train the same linear-elastic model as
before. Strength parameters do not enter the loss, the loss at SRF = 3 is
bit-for-bit the loss at SRF = 1, and the run would produce no FOS. It would
burn 16 GPU-hours and answer nothing.

**I deliberately did not wire it in.** `STEP5_STATUS.md` flags soft-constraint
vs return-mapping as a supervisor conversation, and it changes what the FOS
*means*: the soft version makes the reported FOS depend on the penalty weight,
with no flow rule, hence no dilatancy and no associated/non-associated
distinction. That is a methodological commitment, not a bug fix, and it should
not be made silently inside a debugging pass.

### 3b. The compute is not here

Decided production config (`configs/production_labpc.args`): 8x64,
N_PDE = 10,000, **100,000 epochs**, `--device cuda`.

Measured on this container (1 CPU core, float64, 8x64), 12-30 epoch timings:

| N_PDE | s/epoch |
|---|---|
| 500 | 0.456 |
| 1,000 | 0.842 |
| 2,000 | 1.379 |
| 4,000 | 2.508 |

Linear fit, R^2 = 0.998:

```
s/epoch = 0.2167 + 5.757e-04 * N_PDE
```

Extrapolated to production N_PDE = 10,000: **5.97 s/epoch**, against your
measured 0.573 s/epoch on the RTX 4060 Ti. A **10.4x** slowdown.

| epochs | CPU here | lab GPU |
|---|---|---|
| 10,000 | 16.6 h | 1.6 h |
| 40,000 | 66.4 h (2.8 d) | 6.4 h |
| 100,000 | **165.9 h (6.9 d)** | 15.9 h |

Seven days of continuous compute in a container whose filesystem resets
between tasks. Not survivable, and a shrunken config reported as "the full
run" would be worse than no number.

**What did run:** the training loop end-to-end on CPU at N_PDE = 500 for 30
epochs. All six active terms present, balancer updating, `pde_richards` alive
in the active set, checkpointing and logging fine. The pipeline is healthy --
it is the physics content that is missing, not the plumbing.

**One trap worth knowing.** `train.py`'s closing extrapolation block prints
"100,000 epochs -> 12.66 h" from a run at N_PDE = 500. It is extrapolating in
epochs at whatever N you gave it, which is 1/20th of production. Do not read
that line as a production estimate.

---

## 4. Next session, in order

1. **Close B-1's paper trail.** The regenerated `derived_raw.json` matches to
   2e-15, but `data/step_1_2_1_3_adopted_parameters.json` is a separate
   downstream artefact and has not been re-checked against it.
2. **The yield decision.** Soft constraint vs return mapping, with the
   supervisor. Everything in Steps 5 and 6 is behind it.
3. **Then** wire `pde_yield` into `total_loss` -- balancer weight, L0 baseline,
   a golden-log-keys update, and a test.
4. **Then** `ssr_sweep.py --smoke` on the lab PC, `--w-yield` at 0.3/1/3 to
   show the plateau, before any 100k run.

Add a `tests/test_scripts_import.py` that imports every file in `scripts/`.
All three bugs here would have been caught the day they were introduced.