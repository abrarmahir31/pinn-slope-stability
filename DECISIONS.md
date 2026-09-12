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


**Amended 13 Aug (see D-3.3.1).** `mechanical_residual` now exists, in
`src/mechanics.py`. The `sigma_0` dependency was resolved by *injection*, not
by removal: the function takes `sigma0_star` as an argument and defaults to
assuming it is equilibrated. That assumption is still unverified against the
0.20-0.26 `rho*g` imbalance, so the substance of this decision stands —
`total_loss(include_mechanics=True)` should keep raising until Fix 6 is done.
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

#### D-S.4 REVISED (Day 27) — the premise was about coefficients, not terms

The decision above is **retained** (`normalise="none"`) but its stated reason
does not survive measurement and must not go into Methodology as written.

The argument was: `Pi_R_grav / Pi_R_diff = L_ref/H_ref = 1.030`, therefore "no
term dominates INSIDE the Richards residual". That conflates the Pi groups
with the terms they multiply:

    storage     C*(psi*) dpsi*/dt*
    diffusion   Pi_R_diff div*[K* grad* psi*]
    gravity     Pi_R_grav (dK*/dpsi*)(dpsi*/dz*)

A 3% agreement between two coefficients licenses nothing about two terms
carrying `K*` and `dK*/dpsi*`, which under van Genuchten with n = 1.2 span six
to nine orders across this domain. `scripts/ds4_term_scales.py` tabulates the
ratios that actually decide it, with all field derivatives set to 1 so that
what is being compared is the coefficient structure alone
(`docs/ds4_term_scales.json`):

    D* = Pi_R_diff K* / C*            diffusion / storage
    G* = Pi_R_grav (dK*/dpsi*) / C*   gravity / storage

              psi = -0.01 m        psi = -3 m (floor)   psi = -161 m (crest)
    unit      D*        G*         D*        G*         D*        G*
    Mk        1.1e-07   4.0e-04    1.4e-09   1.5e-07    8.4e-12   2.2e-11
    Mk_d      1.4e-07   4.9e-04    1.7e-09   1.8e-07    1.0e-11   2.7e-11
    Tm        1.5e-02   1.0e+01    8.8e-07   2.2e-04    4.2e-11   2.0e-10

Bisecting for where each ratio reaches 0.1:

* **Mk and Mk_d: never.** At psi = -0.1 mm, D* is still 4.9e-07 and G* 5.3e-02.
  Neither flux term is O(1) against storage at ANY saturation in the marls.
* **Tm: D\* at psi = -1.6 mm, G\* at psi = -0.39 m.** Both are wetter than the
  domain floor's psi_0 = -3 m, so on the initial condition neither is O(1)
  anywhere in Section 5 either. Tm's gravity term is the only one that
  plausibly becomes O(1) during a run, once the front has wetted it.

And the narrow diffusion-vs-gravity claim does not hold either: `D*/G*` runs
from 2.8e-04 to 0.38 across the domain, never 1.03. It is off by between 2.6x
and 3600x depending on where it is read.

**Why `normalise="none"` is nonetheless kept.** The two alternatives are worse
for reasons that are now measured rather than assumed.

`"gravity"` divides by the constant `Pi_R_grav`, which rescales residual and
gradient together and only relabels the imbalance. That part of the original
argument stands.

`"storage"` divides by `C.abs() + 1e-12`, converting the mixed form to the head
form. It looks attractive on the first metric — on an untrained 8x64 net it
takes the top-1% share of `L_PDE_richards` from 99.3-99.95% down to 64-73% and
the number of points carrying 90% of the loss from 2-3 up to 100-127. That
improvement is an artefact of the epsilon guard. 19.7% of collocation points on
that net sit at psi >= 0, where `C* = 0` exactly, so those points all receive
the same divisor `1e-12` and become a constant-weight block that flattens the
distribution. The loss scales exactly as `1/floor^2` (4.28e+21 at 1e-12,
4.28e+05 at 1e-04, 4.41e+01 at 1e-02); at a floor of 1e-01, where the guard
stops dominating, the top-1% share returns to 73-95% and the across-draw spread
rises from 1.3x to 5.5x. The head form is undefined at saturation — that is why
the mixed form exists — and a guard constant is not a substitute for handling
it.

**What the imbalance actually is.** It is not a scaling problem inside the
residual. Above roughly z = 210 m the equation is `C* dpsi*/dt* = 0` and is
correctly so (`docs/richards_terms_prod02.json`,
`docs/inert_fraction_day27.json`: at a 0.1 m / 30 d reach criterion, 94.7% of a
production draw sits where neither flux mechanism can move information at all).
No choice of divisor changes that, because the terms being divided are
physically zero, not numerically awkward. The flag stays as three re-testable
branches; the decision is now recorded as **retained on different grounds**, and
the sentence "no term dominates inside the Richards residual" must be struck.
---

## Step 3.3

### D-3.3.1 — Stress is tension-positive; `effective_stress_nd` was flipped

`nondim.py` shipped two functions with opposite conventions:

* `effective_stress_nd` returned `sigma - Pi_M_couple*chi*psi` — the
  compression-positive form.
* `mechanical_residual_nd` with its default `e_z=(0,-1)` gives
  `div sigma - Pi_M_body*rho_ratio`, i.e. `div sigma + rho*g` — the
  tension-positive equilibrium equation.

Neither had a call site or a test (`grep` on `step-3.2-loss`: two definitions,
two docstring mentions, nothing else), so nothing had ever pinned the
convention and it was free to choose.

**Adopted: tension-positive throughout the residual layer.** `sigma = D:eps`
with `eps` from displacement gradients is tension-positive by construction;
the alternative requires a minus sign inside the constitutive law, and that
minus gets forgotten exactly once. `effective_stress_nd` now returns
`sigma + Pi_M_couple*chi*psi`. At chi = 1 the whole pore term changes sign:
psi* = +0.5 moves from -0.80932 to +0.80932.

`boundaries.sigma_v_geostatic` / `sigma_h_geostatic` remain
COMPRESSION-positive. `mechanics.sigma0_star_from_geostatic` is the single
place in the project where the flip happens, and it is the only place it
should ever happen.

Pinned by `tests/test_coupling.py::test_bishop_sign_is_tension_positive`.
This is the class of error that trains fine, converges, and produces a
plausible-looking field with the pore pressure stabilising the slope instead
of destabilising it.

### D-3.3.2 — Kozeny–Carman is per-stratum, and Tm is undecided

### D-3.3.2 — Kozeny–Carman applies to the marls, not to Tm

`KC_DEFAULT = KCConfig(enabled=True, per_unit={"Tm": False})` in
`src/coupling.py` is the configuration of record.

**The reason is physical, not numerical.** Kozeny–Carman is a
matrix-porosity law. Finding 6 records Tm as fracture-dominated — 2.2%
porosity at K_s = 3.13e-6 m/s — with Tier-C placeholder van Genuchten
parameters, and notes that any matrix SWCC "describes the pathway water does
not use." Applying KC there would assert that elastically compressing a
limestone matrix closes the fractures that actually carry the flow. That is
unsupported, and plausibly the wrong sign: matrix compression under confinement
can open fracture aperture rather than close it.

**What the numbers say.** Measured against the gravity-equilibrated sigma_0
(`sigma0_cache.npz`), volumetric strain and the resulting KC factor are:

| unit | n0 | median \|eps_v\| | p99 | KC factor at p99 |
|---|---|---|---|---|
| Mk | 0.4268 | 3.50e-04 | 1.83e-03 | 0.981 |
| Mk_d | 0.3499 | 2.66e-03 | 8.03e-03 | 0.910 |
| Tm | 0.0221 | 3.35e-04 | 7.67e-04 | 0.898 |

Tm needs only eps_v = 7.0e-4 to move K_s by 10%, against 8.2e-3 for Mk_d — it
is roughly 11x more sensitive per unit strain. But at E = 4.264e9 it strains
about an eighth as much, and the two effects very nearly cancel: 0.898 against
0.910. An earlier draft of this decision claimed Tm would dominate the
feedback; that was wrong, and the correction is recorded here because the
conclusion survived for a different reason than the one first given.

So excluding Tm costs ~10% conductivity change in a unit whose SWCC is a
placeholder, and buys a claim that can be defended in the viva.

**The result worth reporting.** The largest strains in the domain are in
**Mk_d** — median 2.66e-03, nearly 8x Mk's — because at E = 3.806e7 it is the
softest unit by a factor of 5.5. Mk_d is also the unit that fails
(sigma_ci 4.29 vs 17.9 MPa). The porosity–strain feedback is therefore
strongest exactly in the weak zone the failure surface runs through, which
makes two-way coupling load-bearing here rather than a formality.

Caveat: Mk_d's van Genuchten parameters are borrowed from Mk by analogy
(Finding 2, Tier B). The feedback in the failing unit rests on a measured n0
and a borrowed SWCC, so it belongs in the Step 6.2 sensitivity sweep.

**Step 6.2 needs three arms**, all reachable without editing source:
`KCConfig(enabled=True)` (KC everywhere), `KC_DEFAULT` (marls only), and
`KC_OFF` (one-way). Report per-stratum as well as globally.

**Day 27 addendum.** The fracture-vs-matrix argument above has a second,
independent consequence that belongs in Methodology §3 alongside it. Treating
Tm as a porous matrix gives it dtheta = theta_s - theta_r = 0.0149 (from
n0 = 0.0221), and the capacity-limited rain flux q = min(rain.n, K_s) = K_s
then implies a kinematic wetting front at v = q/dtheta = 18.1 m/day — 543 m in
the 30-day window, crossing the whole 158 m domain in about nine days
(`scripts/inert_fraction.py`). A piston front that fast is a matrix-continuum
artefact of the same kind D-3.3.2 excludes KC for. Cite both together; they
are one modelling caveat, not two.

### D-3.3.4 — L_BC_mech inherits L_BC's normalisation, which is the sampler's choice

Squared residuals accumulate across segments and divide by the GLOBAL weight
sum, so each segment contributes in proportion to its point count. With
`sample_boundary` allocating n, n//4, n//3, n//2 by segment, the relative
weight of `cut_face` against `natural_ground` is set by the sampler, not by
any physical argument.

Kept for consistency with `L_BC` rather than because it is right. Measured at
the end of Step 3.3, the three traction-free segments are five to six orders
above the three constrained ones and `cut_face` is 13x `natural_ground`, so
the sampler's allocation is currently not what decides the balance — the
physics is. Revisit if that changes, and record the revisit rather than
silently retuning.

Per-segment parts are MEANS and do not sum to the total, the same distinction
as D-3.2.2.
## Step 3.4 (15 Aug) — found while measuring gradient norms
- [ ] Z_TOE = 271.13 but inside_domain ends at z ~ 270.75 (x = 0); the
      transition is sharp, so this is two constants disagreeing by 0.37 m,
      not a noisy trace. boundaries.py:69 now samples pit_floor to
      Z_TOE - 0.5 as a workaround. Decide which constant is authoritative
      and derive the other from it.
- [ ] Was invisible at N=300 (~0.4 expected hits in the bad band) and
      fatal at N=3000. Phase 4 runs N_BC = 2000-5000, so this would have
      surfaced there instead, mid-training.
- [ ] sigma0.attach_sigma0 tolerated the same points silently
      (max_miss_frac=0.01) while L_BC raised KeyError. Align the two
      tolerances; a 1% silent-miss allowance means sigma_0 can be wrong
      on 1% of points anywhere with no signal.

---

## Sampling

### D-Samp.1 (Day 27) — the hydraulic and mechanical PDE losses want opposite point sets

`scripts/train.py:92` builds ONE `coll` from `sample_interior` and passes it to
both `L_PDE` (Richards) and `L_PDE_mech`. `sample_interior` stratifies by
material on shares 0.25 / 0.35 / 0.40, chosen because Mk_d is the unit that
fails mechanically (sigma_ci 4.29 vs 17.9 MPa) and is only 9.7% of the domain
by area. That is the right stratification for `L_PDE_mech`.

It is close to the worst one for `L_PDE`. Measured
(`scripts/inert_fraction.py`, `docs/inert_fraction_day27.json`): at a 0.1 m /
30 d reach criterion, 94.7% of a 10,000-point production draw lands where
neither capillary diffusion nor gravity drainage can move information at all,
and the marls are the deadest part of it — their front advances 8-10 mm in the
whole window. Oversampling Mk_d by 3.6x spends more of the budget there.

`src/sampling_front.py` provides the hydraulic alternative: log-spaced bands in
initial suction (z - Z_WT), which is the variable K*, C*, the diffusivity and
the celerity are all monotone in, plus a 2 m surface band that elevation cannot
see because the rain BC drives psi toward 0 at the ground surface whatever the
elevation. Live fraction at the 0.1 m criterion goes 5.29% -> 37.93%. Against a
400k-point uniform-in-area reference over 10 seeds, neither sampler is
detectably biased but the front sampler's standard error on smooth integrands
is 2.8x to 6.9x smaller.

**NOT ADOPTED YET, and not wired into `train.py`.** Splitting the two losses
onto separate point sets is a real change to what `total_loss` integrates, and
it should wait on the ansatz decision recorded in `docs/open_items.md` (Day 27)
— see D-Samp.2. Recorded here so the measurement is not repeated.

### D-Samp.2 (Day 27) — the sampler does not fix the seed-variance problem, and residual-based refinement would make it worse

Two negative results that must not be rediscovered.

**The front sampler makes `w^[pde_richards]` worse, not better.** Draw-only
spread at N = 1200 goes 469.9x -> 15989.5x
(`docs/seed_variance_decoupled_front_day27.json`). The Day 26 note assumed the
1264x spread came from the live band being under-sampled. It does not.
`scripts/residual_tail.py` shows 2-3 collocation points out of 4000 carrying
90% of `L_PDE_richards` under EITHER sampler, and every top contributor sitting
at psi = -0.06 to -0.47 m where psi_0 should be -30 to -77 m. The cause is the
ansatz: `NearPhysical.forward` is `psi* = psi0* + eps_psi * raw` off a bare
`nn.Linear`, so at eps_psi = 0.3 between 15% and 35% of points initialise at
psi >= 0 — saturated, in a domain that is unsaturated everywhere by
construction (Z_WT = 197 m is 3 m below Z_BASE). The front sampler moves points
to LOW suction, i.e. nearer the psi = 0 crossing, so it loads more mass onto
the van Genuchten singularity. This is an ansatz decision, not a sampler one.

**Residual-magnitude adaptive refinement (RAR/RAD) is the wrong criterion
here.** In the inert zone |R| is the storage term and is LARGER than in the
live band — on prod02, median 1.24e-03 against 1.12e-05. Refining on |R| would
refine into the dead region. The inert points are trivially SATISFIABLE, not
trivially satisfied; their residual is psi drifting in time where nothing
should happen, which is the 150 m IC violation showing up in the PDE term. Any
adaptive criterion must be the flux fraction
`max(|diffusion|, |gravity|) / |storage|` — the same quantity that fixes
`inspect_richards_terms.py`'s miscalibrated verdict labels, so the two items
are one.