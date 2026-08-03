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
