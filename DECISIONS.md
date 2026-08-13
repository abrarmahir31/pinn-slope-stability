# DECISIONS

Choices that are not obvious from the code, with the reasoning that produced
them. If a decision is later reversed, leave the entry and add the reversal
underneath — the fact that it was considered is itself information.

---

## Step 3.2 — loss module

### D-3.2.1 — `L_data` is dropped, not deferred

There is no data-misfit term and there will not be one. Section 5 has no
piezometric time series and no displacement monitoring at the resolution a
data term would need; the only measured quantities are the geometry traces and
the laboratory parameters, and both already enter through the geometry and
`properties.py`. A data term fitted to nothing would either be identically zero
or would fit the initial condition twice.

Consequence: the model is validated against the *observed failure*, not against
an interpolated field. That is a weaker claim and the write-up should make it.

### D-3.2.2 — Diagnostic parts and weighted contributions are different things

Every loss term returns `(scalar, parts)`. Two kinds of entry live in `parts`
and they are **not interchangeable**:

- **Per-tag / per-segment / per-contact entries** (`pde_richards`, `bc_bench`,
  `interface_mk_tm`, …) are per-set *means*. They divide by their own weight
  sum, so a 100-point segment stays comparable with a 400-point one. They are
  diagnostics. **They do not sum to anything.**
- **`contrib_*` entries** in `total_loss` are weighted contributions,
  `w_i * L_i`. They divide by the global weight sum. **`total == sum(contrib_*)`
  exactly**, and `test_contributions_sum_to_the_total` asserts it.

Read `contrib_*` to see what is driving training. Read the per-set means to see
which part of a term is unhappy. Logging `sum(parts.values())` as the loss is
wrong and will be wrong by roughly a factor of three.

Rejected alternative: making the per-set entries globally weighted so
everything is additive. That destroys the comparability that makes them useful
as diagnostics, which is the only reason they exist.

### D-3.2.3 — Interface sets are keyed by contact; Tm is the lower side by construction

`sample_interfaces` returns `dict[str, Collocation]` keyed `"mk_tm"` /
`"mkd_tm"`, matching `sample_boundary`. Tm is always the lower material at the
`z_tm_top` contact, so `Collocation.tag` carries only the UPPER material and
`L_interface` hardcodes `_INTERFACE_LOWER = "Tm"`.

`mat_a` is read from `coll.tag`, not from the dict key. The tag was set once at
sample time and travels with the Collocation; re-deriving it in the torch layer
would reintroduce the geometry round-trip problem, and it would also make
`loss.py` import from `sampling.py`.

Mk|Mk_d is not sampled: it carries no flux discontinuity while the two marls
share `K_s`, `alpha` and `n`.
`test_mk_and_mk_d_are_hydraulically_identical` guards that assumption, and
`test_mk_mkd_contact_is_not_sampled` is the reminder to revisit this if it ever
fails.

### D-3.2.4 — `cut_face` is currently unenforceable, and that is a finding

Under the initial profile the cut face sits at psi between roughly -74 and
-135 m. At `n = 1.2` the relative conductivity there has collapsed far enough
that its BC residual is O(1e-28) — twenty-eight orders below the segments that
matter. Whatever condition is imposed there is numerically inert at t = 0.

It ships as no-flow (`SEEPAGE_FACE_MODE == "noflow"`), which is harmless
precisely because nothing can be enforced anyway. When the mode changes to
complementarity, the condition will remain inert until the wetting front
arrives and locally raises K, then activate abruptly. That is hard to debug if
it is not expected.

`test_seepage_face_is_currently_no_flow` fails the day the mode changes.

Thesis consequence: combined with finding 1 (the rainfall surface admits
0.02%), the routes by which water can reach the failure surface at t = 0 are
essentially the Tm contact and the far-field condition. The ground surface and
the cut face are both closed.

### D-3.2.5 — Mechanics is absent, not half-implemented

`total_loss(include_mechanics=True)` raises `NotImplementedError`.

`mechanical_residual` needs an equilibrated `sigma_0` from the gravity warm-up,
which **is** Step 3.4. Step 3.2 therefore cannot close before 3.4 opens; this
is coupling order, not slippage.

Shipping a partial version would be worse than shipping none. `base`,
`far_field_f1` and `pit_floor` are cheap Dirichlet/roller conditions, but
traction-free on the three exposed segments needs the stress tensor. Half of it
would make `total_loss` look complete while omitting the free-surface condition
on the cut face — the boundary the failure mechanism runs through.

Note for the supervisor conversation: `L_IC` already carries a `u*, v*` term
pinning displacements to zero at t = 0. "Mechanics deferred" means the
*equilibrium residual* is deferred, not every displacement term.

---

## Scales

### D-S.1 — `E_ref = 1e8 Pa` is a marl scale, and Tm is excluded

`u* = u / U_ref = E_ref / E_actual`, so `E_ref` must sit within about an order
of magnitude of the strata that actually deform, or `u*` is not O(1) somewhere.

Adopted moduli: Mk 2.09e8, Mk_d 3.81e7, **Tm 4.26e9**. The span is a factor of
112, so no single `E_ref` puts all three within an order.

Tm is excluded. It is the crystallised limestone basement, effectively rigid;
small `u*` there is physically correct, not a scaling failure, and the failure
mechanism runs through the marls. The geometric mean of Mk and Mk_d is 8.9e7,
i.e. the adopted 1e8. `U_ref` follows as 1.70 m.

Previous value 1e9 gave `u* = 26` for Mk_d — stiffer than every stratum in the
model. `test_E_ref_is_representative_of_the_adopted_moduli` now fails on it,
and `test_tm_is_stiff_enough_to_justify_excluding_it` is the negative control
for the exclusion.

**The Phase-1 dataset's justification string is wrong** and needs correcting:
it says the moduli "span 8e6 to 2.09e8", which omits Tm entirely and names a
lower bound no stratum has. Its conclusion (revise 1e9 down to 1e8) is right.

Rejected: `2.0e8`, recorded in the Step 3.2 handoff. That is `E_rm` for Mk —
the stiffest deforming stratum, not a representative scale. The handoff
conflated "E_ref follows from MR_adopted" with "E_ref = E_Mk".

### D-S.2 — `nondim.py` is bound to the Phase-1 dataset by test, not by import

`properties.py` reads the Phase-1 JSON as the single source of truth for
material parameters, but nothing bound the JSON's
`nondimensionalisation.scales` block to `nondim.py`. They drifted in **opposite
directions**: `H_ref` ahead in the code (30 → 165, deliberate, so `psi*` fills
[-1, 0] over the IC), `E_ref` behind (still 1e9 against a revised 1e8).

An unbound pair drifts both ways and nothing notices. `nondim.py` stays the
runtime source of truth — importing the JSON at runtime would put a file read
in the hot path — and `test_nondim_matches_the_phase1_dataset` provides the
binding. Every difference must appear in `KNOWN_DIVERGENCES` with a written
reason; anything else fails with both numbers printed.

`test_known_divergences_are_still_divergent` forces resolved entries out of the
allowlist, so it cannot become a place drift hides.

Outstanding: update the dataset's `H_ref_m` to 165 and its `groups` block
(`Pi_R_hz` there is 0.176 against a live 0.971), then delete the
`KNOWN_DIVERGENCES` entry.

### D-S.3 — `materials.C_star` takes dimensional psi and returns dimensionless C*

`C_star(psi, mat, s)` expects `psi` in **metres** — `vg.C` returns `dtheta/dpsi`
in 1/m — and applies `H_ref / dtheta_ref` to the result. So the argument is
dimensional while the return value is not.

Nothing in the name says so, and the `s=None` default makes the scaling look
optional. Passing `psi*` instead of `psi` silently evaluates the retention
curve near saturation and overestimates C* by three to four orders, with a
ratio that varies point to point (it tracks position on the SWCC), so it does
not look like a scale-factor bug.

The convention is currently enforced by one call site and a docstring. Worth
either renaming to make the unit explicit or asserting the magnitude on entry.
### D-S.4 — Step 1.6 scaling closed as `normalise="none"`

`Pi_R_diff = 1.4948e-3` and `Pi_R_grav = 1.5401e-3`; their ratio is exactly
`L_ref/H_ref = 1.030`. Capillary diffusion and gravity drainage are balanced to
3%, so no term dominates *inside* the Richards residual and a constant divisor
has nothing to correct. Both sit ~3 orders below the O(1) mechanical groups,
but that is a cross-equation weighting problem for the `total_loss` weights and
GradNorm; dividing inside the residual rescales the residual and its gradient
together and only relabels the imbalance.

`T_ref = 1 day` is kept — the physically meaningful scale for a rainfall
transient. Forcing `Pi_R_diff = 1` needs `T_ref = L^2*dtheta/(K*H) = 669 days`
against a 30-day window, which abandons the phenomenon the model exists to
resolve. Over 30 days a pressure signal diffuses ~4.5% of the pit depth: small,
not negligible.

The `normalise` flag is retained (`"none"` / `"gravity"` / `"storage"`) because
the branches are cheap and the choice should stay re-testable.

**Why this was deferred so long:** the decision was blocked on a six-orders-of-
magnitude imbalance that stopped existing when `H_ref` went 30 -> 165 m.
`Pi_R_diff` carries `H_ref` linearly, so raising it lifted diffusion onto
gravity — but three docstrings, `validate_nondim.py` and the Phase-1 caveat all
still quoted the pre-change numbers (`~3e-6`, "six orders below storage", a
916-year timescale computed as `L^2/K` without `H` or `dtheta`). Corrected in
Step 3.2.
