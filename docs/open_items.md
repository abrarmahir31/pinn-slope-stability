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
