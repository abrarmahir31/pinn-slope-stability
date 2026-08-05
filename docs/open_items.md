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
