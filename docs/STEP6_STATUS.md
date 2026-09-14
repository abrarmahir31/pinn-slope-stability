# Step 6 status — Days 49–52

Companion to `STEP5_STATUS.md`. Read that one first; this builds on it.

---

## Short version

Days 49–52 all consume FOS. Step 5 doesn't produce one yet, so none of these
can run — same blocker, one layer further down.

Two things I found that are worth knowing regardless of when the blocker
clears: the plan as scheduled needs about five days of continuous GPU compressed
into four calendar days, and one of the numbers in it appears to be pre-written.

---

## 1. The compute doesn't fit, and one arm is far worse than the rest

Costed against your own logged rate (0.6466 s/epoch, day-28 coupled run) and the
sweep driver's 1500-epoch fine-tune over ~28 SRF points:

| | sweeps | hours |
|---|---|---|
| 6.2 OAT (K_s ×2, rain ×2, GSI ×2) | 6 | 45 |
| Day 51 coupling (KC-all, KC-off) | 2 | 15 |
| Day 52 N_PDE levels | 3 | 23 |
| Day 52 seed repeats | 4 | 30 |
| **total, MC only** | **15** | **113 h ≈ 4.7 days** |
| **total, MC + GHB** | **30** | **226 h ≈ 9.4 days** |

Four calendar days are allocated. That's before any re-bracketing, and before
the `--w-yield` 0.3/1/3 plateau runs that Step 5 needs to make any single FOS
reportable.

**The N_PDE arm is the real problem.** `train.py` defaults to `--n-pde 500`. The
plan proposes 5k–50k, so per-epoch cost rises roughly linearly:

| N_PDE | vs. baseline | one sweep |
|---|---|---|
| 5,000 | 10× | ~3 days |
| 10,000 | 20× | ~6 days |
| 20,000 | 40× | ~13 days |
| 50,000 | 100× | ~31 days |

That single line — "N_PDE = 5k/10k/20k/50k" — is about 50 GPU-days on its own,
allocated one day alongside the seed study. Worth checking what `n_pde` your
production run actually used; if it was already 5k the picture improves, but the
scaling is the point and the 50k arm is not survivable either way.

If the mesh-convergence answer has to exist, the cheap version is to run the
N_PDE ladder at a *fixed* SRF near the critical value and show the loss and
admissible-fraction converge, rather than re-running the whole bisection at each
level. You lose a direct FOS-vs-N_PDE curve and gain a defensible convergence
claim for about a fifteenth of the cost. That's a supervisor conversation.

## 2. The 21.5% figure

The Day-51 row says this is where you publish the "one-way overestimates FOS by
up to ~21.5%" number. I went looking for where it came from.

It is not in the repo as a coupling result. There is no recorded one-way vs
two-way comparison anywhere — `DECISIONS.md:355` specifies the experiment but
records no outcome. The only `21.5` in the whole project is here:

```
data/phase1_reference_state.json
  /strata/underclay/rho_note -> "γ_sat=18.4, w=21.5%, γ_dry=15.14 kN/m³"
```

That's the underclay's gravimetric water content. Possibly coincidence and the
figure came from literature I can't see. But either way the structural problem
stands: an experiment whose result is written into the plan before it runs isn't
an experiment. If the coupled sweep comes back at 6%, or negative, the honest
report is 6% or negative — and having "~21.5%" sitting in the schedule makes
that harder to write than it should be.

`coupling_effect()` in the new module signs the result explicitly and its
docstring says what a negative value would mean, because a feedback that
*stiffens* the response is the more interesting of the two findings and
shouldn't get quietly reported as a magnitude.

I'd strike the number from the plan and leave the row as "measure and report."

## 3. Two smaller things

**Day 51 wants three arms, not two.** `DECISIONS.md:355` specifies
`KCConfig(enabled=True)`, `KC_DEFAULT` (marls only) and `KC_OFF`, with
per-stratum reporting. `KC_DEFAULT` is your configuration of record, so a
two-arm one-way-vs-two-way comparison contrasts the baseline against neither of
the alternatives it was chosen over. `coupling_arms()` returns all three.

**The rainfall arm will probably come out flat.** Finding 1 in `open_items.md`
records 0.02% of incident rain admitted, with the flux cap binding on >99% of
points. If that still holds, 10/20/40 mm/hr are numerically the same boundary
condition and the tornado bar has zero span. That's a real result and worth
reporting — but decide now that you'll report it. A flat bar found on day 50
reads as a failed experiment; the same bar predicted on day 49 reads as
independent confirmation of Finding 1.

---

## What I built

`src/sensitivity.py` and `tests/test_sensitivity.py` — 31 tests, suite now 79.

- `E_rm_hoek_diederichs()` and `derive_stratum()` — the GSI arm's full
  derivation chain. Reproduces every stored modulus in the Phase-1 dataset and
  the Tm addendum to four significant figures.
- `oat_arms()`, `coupling_arms()` — the arm lists, with the base level solved
  once rather than three times.
- `tornado()` — bar assembly, ordered by span.
- `seed_statistics()`, `convergence_check()`, `coupling_effect()` — Day 52.

**On the GSI arm specifically.** Moving GSI has to move m_b, s, a *and* E_rm
together. The common half-measure is to re-derive the strength triple and leave
the modulus at baseline — and it fails in a direction that matters here: GSI
40 vs 60 moves marl E_rm by a factor of about 2.9, the softer mass strains
more, and Kozeny-Carman keys off volumetric strain. Holding E_rm fixed
suppresses exactly the coupling Day 51 measures, and the two experiments end up
contradicting each other for reasons that aren't physics.
`test_gsi_arm_moves_modulus_and_strength_together` is the guard.

None of this produces a FOS. It derives the inputs each arm needs and aggregates
the outputs once they exist. The blocker is still the yield term in `loss.py`.