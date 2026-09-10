# Open items

## Blocks Step 3.2
- [ ] No van Genuchten params for Mk, Mk_d, Tm — Rosetta only ran on
      soil-like units. Richards residual needs C(psi), K(psi) everywhere.
- [ ] K_s for marl: field "≈0" (1e-9 assumed) vs any texture estimate.

## Blocks long training runs
- [ ] Z_BASE = 200 m is a modelling choice. Sensitivity test at
      200/170/140 on geostatic stress AND seepage near toe.
      Changes BOUNDS z_max -> invalidates checkpoints.
- [ ] Mk pinch-out at x~90.7: does Mk_d rest directly on Tm, or is
      there an undigitised Mk/Mk_d contact? Check fig19d_section5.png.

## For the methods section
- [ ] X_MK_DIVIDE = 90.7 is the midpoint of the 90.15/91.26 trace
      handoff, not an assumption. Update the code comment.
- [ ] Underclay/slope debris E' are literature values, not site data.
- [ ] Fault gouge clay=0 -> implausibly high K_s. Harmless if not in
      the seepage domain; confirm.
- [ ] Rosetta dry-BD vs moist-BD choice: order-of-magnitude spread in
      K_s. Belongs in Phase 6 sensitivity.

## Pending hardware
- [ ] Lab PC: full config on CUDA, `on cuda: True`.

## Step 1.6 (reopened 4 Aug)
- [x] dtheta_ref = 0.33 provenance RESOLVED (9 Aug). It is Mk's
      theta_s - theta_r = 0.38 - 0.05. The 0.377 in the earlier note
      came from pairing Mk's theta_s (0.38) with Tm's theta_r (0.005);
      theta_r is per-unit (Mk 0.05, Mk_d 0.05, Tm 0.005), not global.
      No change to nondim.py. Perturbation test: setting dtheta_ref to
      0.377 scales both Pi_R groups by 1/k and L_PDE by 1/k^2
      (0.33/0.377 = 0.87533; loss 9.3928e-05 -> 7.1968e-05 = 0.87533^2).
      Confirms the constant is load-bearing through to the loss.

## Step 3.2 design (settled 4 Aug)
- residuals.py provides the AUTOGRAD layer only; the residual algebra
  lives in src/nondim.py (richards_residual_nd, mechanical_residual_nd,
  effective_stress_nd). Do not duplicate.
- C_star convention: C(psi) returns C * H_ref (dimensionless).
- theta_s := n0 per unit (0.427 Mk), overriding the literature 0.38.
- [ ] step21_geometry/*.py use bare sibling imports (import geometry, import vg).
      Only work when run from inside that dir. Needs sys.path shim or package
      conversion before loss.py imports boundaries.py.

## Step 3.2 (6 Aug) — conventions confirmed against code
- CORRECTION to the 4 Aug entry: C_star = C_phys * H_ref / dtheta_ref, NOT
  C_phys * H_ref. Both Richards Pi groups carry the same 1/dtheta_ref divisor
  (Pi_R_diff = 2.7178e-04, Pi_R_grav = 1.5401e-03), so the storage term must
  too. materials.py line 84 already does this correctly. Verified 6 Aug.
- vg.py is the authoritative SWCC; materials.py wraps it for units. Not a
  duplicate implementation. residuals.swcc_from_material is the only place
  physical units enter the residual layer.
      100 MPa (no stratum is near 1 GPa). Gives U_ref = 0.17 m not 1.7 m.
      Does not affect Richards; MUST be settled before the mechanical residual.
- Hydrostatic Richards test: residual 6.78e-21 against terms of 5.94e-04,
  relative 1.1e-17. Mutation-tested against gravity sign flip, missing H_ref
  in Pi_R_diff, and dropped dK/dpsi term - all three caught.

## Step 2.2 mechanical checks (8 Aug, found after repo import)
- [ ] 13_mech_checks.py prints MECHANICAL CHECKS FAILED but exits 0.
      Fix the exit code, or the && chain silently continues.
- [ ] [FAIL] body force in Mk_d: -16900 N/m3. That IS Mk_d's gamma
      (16.9 kN/m3), so the assertion's expected value is wrong, not the
      code. Check whether it compares against rho_dry*g vs gamma_nat.
- [ ] [FAIL] sigma_v at crest: 156022 vs 142412 Pa over 8.4 m (+9.6%).
      Implies gamma_eff 18.6 vs 17.0 kN/m3. Suspect wrong unit assigned
      near the crest, or h measured to a contact not the ground surface.
- Both are in the mechanical IC -> feed the gravity warm-up -> block the
  mechanical residual. Not blocking sampling.py.

## Duplicated constants (8 Aug)
Two bugs today traced to the same cause: geometry.py held its own rounded copy
of a constant that properties.py defines authoritatively.
  - geometry.RHO Mk_d 1723.0 vs properties.RHO_DRY 1722.7 -> 2.94 N/m3 body
    force error, and 25 Pa in sigma_v. Fixed: geometry.py now imports RHO_DRY.
  - boundaries.E_RM held pre-MR-correction marl stiffness (2.286x high). Dead
    constant, deleted.
  - vg.py existed twice; the step21_geometry copy lacked the saturation guard
    and would produce NaN gradients. Deleted.
- [ ] STILL SHADOWED in geometry.py, all also in properties.py:
      K_S, SIG_CI, GSI, M_I, D_DIST, C_RES, PHI_RES.
      In sync as of today (SIG_CI verified identical). No test guards them.
      Same drift risk. Convert to imports from properties.py.

## Step 3.2 (9 Aug)
- residuals.swcc_from_material called _m.C, which does not exist
  (materials exports C_star). The Material->SWCC branch had never
  executed: every prior test passed an SWCC directly. Fixed b5cb629,
  guarded by tests/test_residuals.py. New bug category: untested
  WIRING between two well-tested modules, not an untested function.
- src/nondim.py sat uncommitted-modified across a session boundary
  (dtheta_ref 0.33->0.377, and report_scales labels reverted to the
  4 Aug T*K/L^2 form). Found by `git stash` + re-run. Check
  `git status --short` before trusting any measured number.
- [ ] Tm mechanical params (sigma_ci 70 MPa, GSI 60, m_i 12) are
      literature values for crystalline limestone. Ulusay Table 3b has
      no row for this unit. Confirmed intentional (properties.py:5-6,
      Tm postdates the Phase-1 dataset). Must be declared in methods
      and carried as a Phase 5 sensitivity variable.

      ## Step 3.2 (10 Aug)
- [x] config.BOUNDS covered elevation 0-200 m, i.e. below the domain
      floor, so PINN._scale mapped z to [+0.99, +2.65] and x to
      [-1, -0.23]. Input normalisation was actively making conditioning
      worse. Now a cached snapshot of bounds_from_geometry(), enforced
      by tests/test_bounds.py.

      ## Step 2.2 mechanical checks (closed 10 Aug)
- [x] 13_mech_checks.py exit code fixed; script now gates properly.
- [x] Mk_d body force: assertion's expected value was wrong, not the code.
- [x] sigma_v at crest: 142388 vs 142388 Pa exact (was 156022 vs 142412,
      +9.6%). The computed value was wrong -- unit misassigned near the
      crest. Step 2.2 complete; mechanical IC no longer blocks the
      gravity warm-up.

      ## Step 1.6 (closed 10 Aug)
- [x] DECISION: Richards residual scaling. RESOLVED, and the premise was
      wrong. H_ref = 30 m was the Section 7 confined head above the coal
      seam — a unit absent from Section 5, set before the Day-13
      stratigraphic correction. Section 5's head scale is the hydrostatic
      IC range (psi_0 = Z_WT - z, ~0-161 m), so H_ref = 165.
      Pi_R_diff 1.4948e-03, Pi_R_grav 1.5401e-03, ratio L/H = 1.0303.
      Both terms O(1) with T_ref = 1 day retained. No division-through,
      no T_ref = L^2/K.
- [x] CHECK 2 "rebalanced" block: superseded. L/H was never a geometric
      constraint, just a mismatched reference pair.
- [x] dtheta_ref provenance: it IS the literature Mk theta_s 0.38 - 0.05.
      Now derived from properties.THETA_S/THETA_R rather than a literal.
      The 0.377 figure in the old note (theta_s := n0) was stale.

## OPEN (2026-08-13, high) — Bishop pore term is absolute, and probably wrong-signed

At the true initial state (u=v=0, psi=psi_0, equilibrated sigma_0) the
mechanical residual must vanish. With bishop=False it does. With bishop=True
it gives |res_z| median 0.604 (Mk), 0.579 (Mk_d) -- ~20% of Pi_M_body.

Cause 1: the pore term uses absolute chi*psi. sigma_0 is a TOTAL stress, so
the increment chi*psi - chi_0*psi_0 is what belongs there. Verified: with the
increment the residual is exactly 0.0 in all three strata. chi_0*psi_0 must
enter DIFFERENTIABLY -- the term acts through div, so a detached psi_0 changes
nothing.

Cause 2: D-3.3.1 flipped effective_stress_nd to sigma + Pi*chi*psi. Tension
positive, sigma' = sigma + chi*p, so total = sigma' - chi*p. D:eps is
effective; equilibrium acts on total; the residual should SUBTRACT. The Fix 1
formula is a correct total->effective converter used in the wrong direction.

Do not run coupled training until both are resolved. Blocks the traction-free
BCs, which are a condition on TOTAL stress.

- [ ] D-W.2 anchor is fixed at t = 0 but bc_mech is not the largest-
      gradient term during training (pde_mech reaches 4-5x it). pde_mech
      ends 3.66x worse. Not coupling -- include_feedback=False gives
      4.19x. geomean is worse on every term. Try argmax-g anchor
      re-selected at each update, or SA-PINN pointwise weights.

      ## Day 24 findings

**L-BFGS absent from Step 3.4.** No `LBFGS`/`lbfgs`/`closure` anywhere in
the repo. The roadmap's two-stage Adam -> L-BFGS protocol is half-built:
Adam works, the second stage was never written. Days 25-27 assume both.
Decide before launch whether to write it first or run Adam while writing.
Open sub-question: does the checkpoint carry balancer state, so L-BFGS can
resume from an Adam run? Check `_atomic_save`'s payload.

**float32 ruled out (closed).** `derivatives.py:56`: float32 bottoms the
hydrostatic Richards test at ~1e-7 vs 1e-14, too coarse to separate
"exactly zero" from "nearly cancelled" -- which would destroy the
interface finding (~1e-26) and the Day 22 diffusion-length argument.
Production stays float64. `check_float32.py` corroborates.

**`KCConfig.is_on` fails open.** An unrecognised material tag returns the
global `enabled` rather than raising (`check_kc_range.py`, tag "XX" ->
1.1104 at eps_v=1e-2). A misspelled stratum name in `L_PDE` would silently
receive KC feedback. Consider validating against `load_materials()` keys.

**`L_IC` edit pending.** `grad_norm_table.py:102` carries
`# needs the L_IC edit`. Resolve or confirm already done.

**Tm exclusion for Methodology §3.** KC_DEFAULT has `per_unit={"Tm": False}`.
Rationale is D-3.3.2 (`test_coupling.py:363`): Kozeny-Carman is a
matrix-porosity law, Tm is fracture-dominated (n0=0.0221, K_s=3.13e-06,
three orders above Mk). Quantitative backing in `check_kc_range.py`.
Also record `kozeny_carman_factor`'s keyword defaults there.
## Day 26 findings

**Fresh clone does not pass `pytest -q`.** `src/step21_geometry/ic_cache.npz`
and `sigma0_cache.npz` are gitignored, and 11 tests fail / 42 error without
them. Regenerate with `cd src/step21_geometry && python 14_ic_checks.py` and
`python 17_sigma0_solve.py` (both self-check and print ALL CHECKS PASSED).
Belongs in README.md, not here.

**`pi_seed` was missing the ansatz prefactors (fixed).** The seed is
`((Pi_M*eps_uv)/(Pi_R*eps_psi))^2 = 4.221e+01`, not `(Pi_M/Pi_R)^2 =
3.799e+06`. What multiplies the network inside a residual is the Pi group AND
the eps that `NearPhysical` puts in front of the raw net, and `grad L =
(2/N) sum r grad r` carries it twice. Measured check: the cross-equation
requirement is w^ = 6.93e+02 against pde_mech, 3.47 against bc_mech; the new
seed leaves c = 16.4 and 0.082, the old one 1.8e-04 and 9.1e-07. The residual
16x is van Genuchten K_r nonlinearity, which is not a prefactor and is the
part the balancer is supposed to measure.

Why it survived eleven days: at `eps_psi = 3e-3` the omitted ratio
`eps_uv/eps_psi` was 1/3, so the seed was wrong by 9x and the correction
absorbed it silently. At 0.3 the same omission is 9e4.

**CORRECTIONS to the Day 25 note.**
- `pde_mech`'s grad-norm ratio between the two ansaetze is 0.98634, not
  exactly 1.0. It moves because `total_stress_star` carries the Bishop
  increment `chi*psi - chi_0*psi_0`. Small, but it is the two-way coupling
  being live; do not write "exactly 1.0" into Methodology.
- The note does not mention that `bc_mech` -- the anchor -- moved 1153.6x.
- `grad_norm_table.py:63`'s "stale BOUNDS" item is CLOSED. `src.config.BOUNDS`
  agrees with `bounds_from_geometry` to 6.4 cm on every component and is
  pinned by `tests/test_bounds.py::test_bounds_literal_matches_geometry`. The
  x_max 4.41 / z_max 1.18 figures in the note match nothing in the tree.
- The per-segment `parts` dicts in `loss.py` are detached BY DESIGN (D-W.5,
  reported not weighted). A zero gradient column off `per_segment=True` is
  the measurement being taken wrong, not a bug. Measure one segment at a time
  by passing a one-segment `bcs` dict.

**THE ANCHOR IS MEASURING THE HYDRAULIC FIELD.** `L_BC_mech` split by segment
at 8x64, N=300 (`||grad L||`):

    segment          kind          eps_psi=3e-3   eps_psi=0.3     ratio
    cut_face         traction        7.6392e-03    8.7573e+00    1146.4
    natural_ground   traction        1.0026e-03    2.9407e+00    2933.0
    bench            traction        2.4729e-03    1.3619e+00     550.7
    pit_floor        constrained     1.6396e-04    1.6396e-04       1.0
    base             constrained     1.0379e-04    1.0379e-04       1.0
    far_field_f1     constrained     1.0291e-04    1.0291e-04       1.0
    <total>          --              1.9462e-03    2.2451e+00    1153.6

The three constrained segments are bit-identical -- they constrain u*, v*
only. All movement is in the traction-free segments, which use
`total_stress_star` and therefore carry the pore increment; with
`eps_uv = 1e-3` the `D:eps` part of those tractions is negligible, so what is
left is `sigma0.n` plus pore. Isolating it by rebuilding at `eps_psi = 0`
(pore increment identically zero): `||grad L_bc_mech|| = 1.9038e-03` against
2.2451e+00 at 0.3. **0.085% of the anchor's gradient is mechanical.**

D-W.2 pinned `bc_mech` at 1.0 on the premise that it is a stable mechanical
reference holding the largest gradient. Neither half holds at 8x64 under the
new ansatz, and every w^ in `PI_SEED_8X64` is normalised against it. D-W.2 was
settled at 2x64 and has still never been rerun at 8. THIS IS THE DECISION THAT
GATES THE PRODUCTION RUN.

Candidate targets, required w^ at eps_psi = 0.3 (spread g_max/g_min = 9.42e6,
i.e. 6.97 orders -- wider than a +/-3 clip can span from a w0 = 1 seed, so
only the SEED can carry it, whatever the target):

    term            g_i        anchor=bc_mech  anchor=pde_mech   geomean
    pde_mech        4.487e+02       5.004e-03        1.000e+00  6.671e-04
    ic_head         4.410e+00       5.090e-01        1.017e+02  6.787e-02
    bc_mech         2.245e+00       1.000e+00        1.999e+02  1.333e-01
    bc              1.589e+00       1.413e+00        2.823e+02  1.883e-01
    pde_richards    6.478e-01       3.466e+00        6.927e+02  4.621e-01
    interface       9.882e-04       2.272e+03        4.541e+05  3.029e+02
    ic_disp         4.762e-05       4.715e+04        9.423e+06  6.286e+03

**`ic_disp` starts outside the clip and will stay there.** Its gradient norm
is byte-identical under both ansaetze (4.7618e-05; it has no psi dependence),
so w^ went 4.09e+01 -> 4.71e+04 purely because the ANCHOR grew 1154x. Held at
w0 = 1 it needs 4.67 orders and the balancer pins it at c = 1e3 -- prod01's
`bc` failure relocated to a new term. Confirmed in the smoke run:
`clipped: ["ic_disp"]` from step 100 onward. Geomean improves it to 3.80
orders, still outside. Seeding it at 4.71e+04 is the other option and is NOT
obviously wrong -- `ic_disp` is a t=0 constraint, so enforcing it hard does
not stop u,v evolving at t>0 -- but it means the optimiser spends as much
effort on an already-satisfied IC as on `pde_mech`. Decide deliberately.

**`activity_floor = 1e-12` cannot see this class of problem.** It catches
terms that are numerically zero, not terms that are small because they are
satisfied. At eps_psi = 0.3, `ic_disp` sits at 1.06e-07 of g_max and
`interface` at 2.20e-06, both far above the floor and both being driven
toward a parity they should not have. A floor of ~1e-5 would freeze exactly
those two by measurement rather than by the hand-written `hold` list in
`seed_from_gradnorms`. NOT done: `pde_richards` sits at 1.44e-03 of g_max, so
a 1e-5 floor leaves it only 144x of headroom, and dropping out of the active
set is precisely the Day 25 disaster. Needs a regime-aware rule (freeze on
p ~ 0.5, not on magnitude), not a bigger constant.

**`inspect_prod01.py` silently misread prod01 (fixed).** It rebuilds
`NearPhysical` before `load_state_dict`, so it inherited the new default
`eps_psi = 0.3` and would have reported prod01's psi* 100x too large -- the
opposite of the collapse it exists to detect. Now takes a checkpoint path and
`--eps-psi`; pass `3e-3` for anything trained before 2 Sep.

### Day 26, later: the measured seed was never a measurement

`scripts/seed_variance.py` (new) varies ONLY the collocation draw and
recomputes the `w^` column. 8x64, eps_psi = 0.3, t = 0, three sampling seeds.
Artifact: `docs/seed_variance_day26.json`, pinned by
`test_the_measured_pde_seeds_are_not_reproducible_across_draws`.

    term            spread across 3 draws        verdict
                     N = 300     N = 1200
    pde_richards        2.0x      1264.1x        unusable
    pde_mech           45.0x        30.1x        unusable
    interface           9.1x         1.4x        --
    ic_disp             3.4x         3.1x        stable
    bc                  2.5x         2.1x        stable
    ic_head             2.4x         2.2x        stable
    bc_mech             1.0x         1.0x        anchor by definition

`w^[pde_richards]` moves 1264x between two draws of the same objective -- more
than the +3 clip can correct. Which draw you happen to seed from therefore
decides whether the run ends up clipped. And the spread GROWS with N, which is
the opposite of Monte-Carlo convergence: it is the signature of an estimator
whose variance is dominated by rare extreme samples. van Genuchten K_r spans
nine orders across this domain (1.7e-04 in Tm near the surface to 2.5e-13 at
the crest), so the squared Richards residual and its gradient are set by
whichever few collocation points land in the extreme cells.

This is the single explanation for three separate failures:
  - the 16 Aug `pde_mech = 2.925e-06` seed going stale;
  - `t0` (w0 = 3.47) driving pde_richards into the +3 clip by step 400;
  - `geomean` (w0 = 115.8) driving it into the -3 clip by step 200.
None of them was a wrong choice between two good numbers. All three were
draws from a distribution with no useful centre.

MATCHED THREE-ARM ABLATION, identical N/seed/trajectory, L/L0 at step 550
(runs/seedablation_{t0,geomean,formula}):

    term            t0        geomean      formula     best
    bc_mech         8.427e-02  5.351e-02   1.550e-02   formula
    pde_mech        2.744e+01  3.088e+01   6.291e+00   formula
    pde_richards    1.614e-03  4.133e-04   1.727e-06   formula
    bc              1.099e-01  3.863e-02   3.068e-02   formula
    ic_head         5.243e-01  1.744e-01   2.943e-01   geomean
    ic_disp         7.190e-07  2.582e-07   7.775e-07   geomean
    interface       2.986e-20  4.375e-23   9.026e-24   formula
    weighted spread     43.79      109.1        23.85  formula

The analytic seed wins five of seven terms and the spread, and it is the only
arm that was not tuned. `--seed-source` now defaults to `formula`; `t0` and
`geomean` are kept as named arms so the ablation stays reproducible.

CAVEAT before this goes in Methodology: one network seed, 550 epochs, N = 1200,
CPU. Replicate at 2-3 network seeds and production N. The direction is strongly
suggested, not established.

WHAT THIS DOES NOT FIX. `ic_disp` is stable across draws (3.1x), so its
4.71e+04 requirement is real and it stays clipped in all three arms. That is
the anchor problem, not the seed problem, and it is still the item that gates
the production run.

Note also that `ic_head` ends ABOVE 1.0 relative to L0 in every arm at N=1200.
Under eps_psi = 3e-3 the IC was satisfied by construction; it is now a real
soft constraint and no seed choice is satisfying it. `inspect_prod01.py` on the
N=2500 smoke checkpoint shows the t=0 field off the analytic IC by up to 84 m
of head at step 500. Watch it in the production run.
