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
- [x] Pi_R_diff was T*K/L^2 (units 1/m). Fixed to T*K*H/L^2.
      Gravity/diffusion ratio now L/H = 5.67, was 170.
- [ ] CHECK 2 "rebalanced" block is stale: under corrected groups,
      T_ref = L^2/K gives Pi_R_diff = H_ref = 30, not 1. No choice of
      T_ref makes both Richards terms O(1); the ratio L/H is fixed by
      geometry. Rewrite the text.
- [ ] CHECK 3 round-trips the MECHANICAL residual only (2342 N/m^3).
      Richards has no round-trip test - and Richards is where the bug was.
- [ ] DECISION: Richards residual scaling. Preferred = divide the whole
      Richards equation by Pi_R_grav so all three terms are O(1) with
      T_ref = 1 day retained. Alternative = T_ref = L^2/K.
- [ ] rho_b_ref = 1800 kg/m3 is an arbitrary reference, not a Section 5
      material (Mk sat = 1946). Either retag the comment or set to 1946.

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
- [ ] dtheta_ref = 0.33 has no recorded provenance. Mk's own theta_s-theta_r
      is 0.377. Now sits in the denominator of both Richards groups, so it is
      no longer cosmetic. Find where 0.33 came from or retag it.
- [ ] E_ref = 1e9 in nondim.py contradicts the Phase-1 decision to drop it to
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
