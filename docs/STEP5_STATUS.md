# Step 5 status — what I found, what I built, what blocks you

Written against the day-28 snapshot (`pinn_repo_day28.zip`, `pinn_logs_day28.zip`).

---

## Short version

The days 39–48 plan cannot run as written, for a reason that has nothing to do
with how much compute you throw at it: **the model has no failure criterion in
it.** Strength parameters do not appear in the governing equations, so reducing
them changes nothing, and an SSR sweep against the current formulation returns a
converged, smooth, meaningless curve at every SRF. It will not crash. That is
the dangerous part.

I did not produce FOS_MC, FOS_GHB, or the failure surfaces. I could not, and
neither could the lab PC in its current state. What I did instead was build the
missing constitutive machinery and test it.

---

## The blocking finding

`mechanics.total_stress_star` assembles

```
sigma = sigma_0 + D:eps - (chi*psi - chi_0*psi_0)
```

Linear-elastic Hooke plus a Bishop pore term. `mechanical_residual` takes its
divergence. A grep across `src/` for any yield criterion — `yield`, `mohr`,
`coulomb`, `plastic`, `tau_f` — returns only the word "yields" in two
docstrings. There is no yield surface, no return mapping, no plastic strain.

So c, phi, m_b and s are not arguments to anything the network minimises. At
SRF = 3 the loss is bit-for-bit the loss at SRF = 1. The "critical SRF where
L_PDE flatlines high and displacement gradients blow up" never arrives, because
nothing was ever constrained by strength in the first place.

This is why Step 5.1 is not a three-day task. It is not the sweep that is
missing — it is the plasticity.

## Three things that also don't line up

1. **No frozen baseline.** The repo zip contains no `.pt`, `.pth`, `.ckpt` or
   any other weights. Step 5.1 fine-tunes "from the frozen baseline"; that
   artefact is not in what you sent. The log confirms a coupled HM run reached
   step 2200 at total loss 1.2e-3 in an L-BFGS phase, so it exists somewhere —
   just not here.

2. **The repo is at day 28, the plan starts at day 39.** `step34_handoff.md`
   has mechanics as the *next* step. The log shows `pde_mech` and `bc_mech`
   live, so days 29–38 happened, but that work isn't in the snapshot. Anything
   I write against this tree may already be stale.

3. **Compute.** This container has 1 CPU core, 3 GB of RAM, no GPU, and not
   enough disk to install torch. Your lab PC logged 0.647 s/epoch. A 28-point
   sweep at 1500 epochs is about 7 hours *there*, doubled for GHB. Here it would
   be weeks, if it could run at all.

Even with all three fixed, finding 1 stands.

---

## What I built

Four files, in `pinn_step5/`. Drop `src/*` into `src/`, `scripts/*` into
`scripts/`, `tests/*` into `tests/`.

| File | Status |
|---|---|
| `src/strength.py` | MC + GHB criteria, both reductions. **Tested.** |
| `src/failure_surface.py` | gamma_max, ridge tracing, band-width check. **Tested.** |
| `src/plasticity.py` | Yield-constraint loss term. **Algebra tested.** |
| `scripts/ssr_sweep.py` | Sweep driver, bracket + bisect + resume. **Written, never run.** |
| `tests/test_strength.py` | 48 tests, all passing. |

The algebra layers import no torch, which is what let me test them on a box
without a GPU — and is also the repo's existing `nondim.py` / `residuals.py`
convention, so it should sit naturally where you put it.

Tests follow your negative-control rule from `step34_handoff.md`: every "this is
zero" claim is paired with a "this is not zero" that would fail if the function
were `return 0.0`. `test_gamma_max_vanishes_under_rigid_rotation` is paired with
`test_gamma_max_is_nonzero_for_simple_shear`; `test_mc_yield_is_zero_on_the_envelope`
with the off-envelope control; and so on.

### On GHB reduction

There is no defensible way to divide m_b and s by the SRF — they are envelope
shape parameters, not strengths, and dividing them reduces strength by an amount
that varies with confinement. `reduce_ghb` implements Hammah et al. (2004)
instead: take instantaneous MC tangents of the original envelope over a
confining range, reduce those the classic way, refit (m_b', s') with `a` held.

**The confining range is a modelling choice and it is yours.** Set it from the
baseline stress field along the candidate failure zone, not the full domain —
the deep Tm points sit at confinements the slip surface never sees.
`test_reduce_ghb_result_depends_on_the_fit_range` exists to stop that decision
from going quiet. Report the range next to every FOS_GHB.

### One thing to check in Phase 1

I could not initially reconcile the stored `hoek_brown_mc_equivalents` against
Hoek's closed-form fit, and spent a while assuming the dataset was wrong. It is
not — it reproduces to four significant figures across all four strata. My error
was using the D = 0 form of `s`. Every stratum is stored at **D = 1**, giving
s = exp((GSI−100)/6), not /9. For marl that is 2.404e-4 against 3.87e-3 — a
factor of four in unconfined rock-mass strength.

Worth knowing because it is silent: the D = 0 form produces a plausible number
and a wrong slope. If any of your Step 1.5 derivations assumed D = 0, they need
rechecking.

---

## What the sweep would need before it runs

1. **Wire a yield term into `loss.py`.** `total_loss` needs a `yield_params`
   argument and a `pde_yield` entry in its parts dict. This touches the
   balancer (a weight for the new term) and the L0 normalisation (a baseline
   value for it). This is the blocking task.

2. **Decide soft constraint vs return mapping.** `plasticity.py` gives you the
   soft version: penalise stress outside the yield surface, and past the
   critical SRF equilibrium and admissibility stop being jointly satisfiable.
   It is cheap and the plateau is a real signature. But the FOS depends on the
   penalty weight, there is no flow rule, so no dilatancy and no correct
   post-peak kinematics, and associated-vs-non-associated does not arise
   because there is nothing to associate. For a frictional material those
   differ materially and reviewers know it.

   Return mapping gives a defensible FOS. It is weeks, not days.

   This is a supervisor conversation, not a coding decision, and it is worth
   having before you spend GPU time.

3. **Then** the sweep: `--smoke` first, `--w-yield` at 0.3/1/3 to show the
   plateau, and at least one FOS re-run with `--cold-start` to check the
   warm-start chain didn't contaminate itself.

---

## On the three hours

I'd push back gently on one thing. "Lock the headline numbers" while you're
away is the part I'd want to talk you out of even if the compute existed and the
plasticity were wired. FOS_MC and FOS_GHB are the results you'll defend; the
judgement calls behind them — where the loss plateaus, whether the admissible
fraction fell as a band or collapsed everywhere, whether the localisation is
physical or a resolution artefact — are exactly what a viva probes. Those need
you watching the sweep, not reading its output.

Building the machinery while you're out is fine. Reading the diagnostics isn't
something I should do on your behalf.

The useful news is that days 39–41 were never going to be "run the sweep"
anyway. They were going to be "discover the sweep can't run, then write the
plasticity." The first half of that is now done.