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

### D-3.3.3 — `effective_stress_nd` removed in favour of `total_stress_nd` — ENTRY MISSING

**This decision was made and implemented; only the entry is absent.** Recorded
here (Day 28) so the gap is visible rather than silent. Previously read as a
numbering error — it is not: `.1 .2 .2 .4` was a duplicated `D-3.3.2` header
with no body (now removed) PLUS a genuinely unwritten `.3`.

Cited by five files, so the decision is load-bearing:
`src/nondim.py:228`, `src/mechanics.py:29`, `src/loss.py:339`,
`tests/test_initial_equilibrium.py:20,87,117`, `tests/test_coupling.py:12`.

What the code attests, and nothing more:

* `effective_stress_nd` is REMOVED, not aliased. It now raises
  `NotImplementedError` (`nondim.py:227`). The stated reason is that both the
  sign AND the argument changed — the argument is now an increment — so a
  silent forward would be "wrong twice over".
* `total_stress_nd` replaces it.
* This is DISTINCT from D-3.3.1, which flipped the sign of
  `effective_stress_nd` but kept the function. D-3.3.3 is the later removal.
* `test_initial_equilibrium.py:117` records the initial-equilibrium residual as
  0.604 / 0.579 / 1.06e-06 BEFORE D-3.3.3.
* `loss.py:339` states the traction-free condition is on total stress and that
  D-3.3.3 is what made that quantity assemblable.

**The reasoning is not reconstructible from the source and has to be written by
hand.** Do not let this stub stand as the entry — `test_initial_equilibrium`
and the `bc_mech` anchor both rest on it.

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

## Ansatz (Day 28)

### D-A.1 — `unbounded` is rejected on the 3×3; cap vs exp is left open

> **Closed by D-5.7 (Day 41):** `cap` was chosen for the production baseline.
> Note the seed conflict recorded there: in the ablation below, `cap` failed
> on seed 20250812, which is the seed in `configs/production_labpc.args`.

`NearPhysical` gains `mode` ∈ {`unbounded`, `cap`, `exp`} and `cap_k`. **The
default is unchanged (`unbounded`)**, so every prior checkpoint, `PI_SEED_8X64`
and Step 3.4 weight stays valid; the arms are selected explicitly by
`--ansatz`.

Measured, 3 arms × 3 seeds × 2000 epochs at N_PDE = 10000, production settings
(`scripts/ansatz_ablation.py`, `docs/ansatz_ablation_prod.json`). L_i/L0_i at
step 2000:

| arm | bc | pde_mech | pde_richards | psi_max (m) |
|---|---|---|---|---|
| `unbounded` | 3.3e-01 .. 1.5e-03 | 1.4e-03 .. 2.7e-01 | 1.0e-05 .. 3.8e-05 | −3.3 .. −22.7 |
| `cap` k=100 | 1.1e-02 .. 3.5e-05 | 2.0e-04 .. 2.3e-04 | 2.3e-05 .. 4.2e-01 | −0.00 .. −3.26 |
| `exp` | 3.1e-05 .. 7.7e-04 | 1.6e-04 .. 4.2e-04 | 2.2e-03 .. 1.2e-02 | −2.76 .. −3.07 |

**`unbounded` is REJECTED.** `bc` is three to four orders worse in every seed,
and it is the only arm that drives `pde_richards`/`pde_mech` onto the clip. The
mechanism is not a tuning failure: the domain is unsaturated everywhere
(Z_WT = 197 m against Z_BASE = 200 m), and `psi* = psi0* + eps_psi*raw` off a
bare `nn.Linear` puts 15–35% of points at psi ≥ 0 at eps_psi = 0.3 — unphysical
by up to 185 m of head.

**cap vs exp is NOT settled.** `cap` is better when it works, but seed 20250812
gives `pde_richards` = 4.18e-01 (reproduced from CPU's 2.56, so not noise) and
is the seed whose `psi_max` pinned at −0.00 against the cap. `exp` is uniformly
mediocre and never fails. One more seed each decides whether cap's failure is
1-in-3 or rarer.

**`unbounded/20250812` is not matched** and nothing may lean on it: it ran with
`--n-iface 400` while the other eight used 500, and it is the only cell with
`ic_head` > 1. Clear and re-run before quoting it.

**Conditional validity.** psi ≤ 0 holds only while
`INFILTRATION_MODE = "capacity_limited"` and `SEEPAGE_FACE_MODE = "noflow"`,
both of which are open items. If either changes, this decision is reopened
rather than adjusted.

**Read `(ic_head, psi_max)` as a pair.** Low `ic_head` with a very negative
`psi_max` is not success — it is the field parking near its IC and doing no
physics. `unbounded` seed 7 ends at `ic_head` 0.30 with `psi_max` −22.7 m;
seed 20250812 at `ic_head` 2.13 with `psi_max` −6.8 m. The second is arguably
healthier.

### D-A.2 — the regime signal is the flux fraction, and it has no global threshold

The blocker D-A.1 exposed is that `activity_floor` gates on gradient-norm
MAGNITUDE, which cannot distinguish a term that is small because it is SOLVED
from one that is small because its equation has degenerated to
`C* dpsi*/dt* = 0`. In the `exp`/seed-7 polish the balancer ended with
`w[pde_richards]` = 7366 against ‖grad L‖ = 4.97e-07, seven orders below
`pde_mech`.

**No constant can fix it, and this is measured rather than argued.** Across all
81 balancer updates in the Day-28 log, `pde_richards` carries a *larger*
gradient than `ic_disp` — relative norms 1.4e-05 to 5.5e-05 against 2.5e-06 to
5.9e-06. Any floor high enough to freeze the first freezes the second, and
`ic_disp`'s 4.71e+04 is a real requirement (D-W.2). The two are interleaved in
the wrong order.

The signal adopted is the pointwise flux fraction
(`residuals.flux_fraction`):

    p = max(|diffusion|, |gravity|) / (|storage| + |diffusion| + |gravity|)

**Freeze LOW, not at 0.5.** p → 0 is the degenerate limit; p ≈ 0.5 is storage
balanced against one flux, i.e. what a solved transient Richards equation looks
like. The Day-27 note's "freeze on p ≈ 0.5" recorded the *healthy* value, not a
trigger.

**But 0.05 is also wrong, and by four orders.** The attainable ceiling on p is a
MATERIAL property, recomputed by `residuals.attainable_flux_fraction`: 4.027e-04
(Mk), 4.922e-04 (Mk_d), 9.118e-01 (Tm). The marls cannot reach 0.05 at any
saturation, so an absolute cut is an unconditional disable of `pde_richards` in
12.7% of the domain however well the network is solving. Tm only exceeds 0.05
within ~10 cm of saturation, while the healthy arms sit at psi_max = −2.7 to
−3.3 m, so it disables Tm in practice too. **Normalise by the per-material
ceiling before comparing across strata, or scope the rule to Tm and record
that.** The reduction across the three tags into one weight for `pde_richards`
is OPEN.

**The threshold cannot be calibrated before D-A.1 closes.** van Genuchten C* is
identically zero at psi = 0, so at a saturated point the storage term does not
shrink — it VANISHES, and p reads 1.000 by construction. Measured, 36.9% of Tm
at t = 30 under `unbounded` (seed 7) sits at psi ≥ 0 reading p = 1.000, while
the physical psi < 0 points sit at 2.48e-05. The equation has changed type at
those points; they are not Richards at all. A threshold fitted there measures
the ansatz.

**Reduce with a fraction, not a quantile.** `frac(p > p_ref)` is bounded and
cannot be dragged by a handful of extreme points; across five collocation draws
on a fixed network the spread is 1.04x against 2.54x for p99. A tail-sensitive
statistic would rebuild the D-W.7b seed-variance failure one layer up.

Implemented as `BalancerConfig.regime_floor`, **default 0.0 = OFF** (D-W.8), so
the magnitude-only behaviour is unchanged until a floor is calibrated.

### D-A.3 — a checkpoint's ansatz is read from the checkpoint, not from a flag

`mode` changes the FORWARD PASS, so rebuilding `NearPhysical` with the default
evaluates a `cap` or `exp` checkpoint as `unbounded` and yields a different
field from the same weights. `train.py` has always stored it (`cfg: vars(a)`);
neither `inspect_richards_terms.py` nor `flux_fraction.py` read it until now.
Both take it from `cfg["ansatz"]`, warn when the key is absent, and accept
`--mode` only as an override. Symptom of the bug: psi ≥ 0 points appearing in a
run of a bounded arm.
---

## Step 5 — SSR decisions (Day 41, supervisor meeting)

Evidence for all entries: `docs/SUPERVISOR_BRIEF.md`,
`scripts/decision_evidence.py`, `docs/results/decision_evidence.json`. Every
evidence number is a POINTWISE admissibility check of the frozen elastic
baseline (`baseline-v1`), not a factor of safety.

**Source labels.** *Decision* = made by the student and supervisor.
*Analysis note* = measurement or interpretation added while recording, not a
decision; it can be challenged without reopening the decision.

### D-5.1 — Soft constraint, not return mapping

*Decision.* The yield surface enters as the penalty
`w_yield * mean_w[relu(f/sig_ref)^2]` (`loss.L_yield`). No flow rule, no
plastic strain, no stress projection.

*Decision, reason.* Implemented and tested now; return mapping is weeks of
implementation and verification. The penalty-weight plateau (w_yield 0.3, 1,
3) is required with every reported FOS.

*Analysis note.* The meeting also raised "hard return mapping breaks automatic
differentiation". It is not recorded as a reason: radial return is
piecewise-smooth and differentiable return-mapping layers exist, so the claim
would not survive examination. The plateau demonstrates that **the FOS** is
insensitive to the penalty scaling over the tested range. It does not show
stresses are admissible; the admissible fraction is reported separately at
every SRF. No dilatancy and no associated/non-associated distinction belong in
Limitations.

### D-5.2 — MC-SSR strength

*Decision: (d) rejected.* Reducing the bedding residual pair c = 4.4 kPa,
phi = 15.4 deg as a continuum strength at every point. Evidence: pointwise on
the baseline at SRF 1, Mk 0.000, Mk_d 0.069, Tm 0.019 admissible (area
0.023). **98% of the slope is already outside the envelope**, the limestone
basement included, before any reduction. A CPU smoke sweep showed
admissibility rising with SRF as the penalty reshaped the field, so no
failure could be detected. This supersedes the Phase-1 dataset note
`mc_strength.design_pair` ("THIS is the pair to reduce in the MC-SSR sweep"),
which is left in place.

*Decision: (b) not selected, for schedule.* Bedding strength carried on a
bedding interface or by a ubiquitous-joint model needs new constitutive code
and its verification, which the development timeline does not allow. It is
**not** rejected on physical grounds; it is the physically closer
representation of a bedding-controlled mechanism, and belongs in Future Work.

*Decision (Day 41, second ruling): (a), rock-mass MC equivalents, with
`--mc-sig3max 400000` (400 kPa; 850 kPa named as the alternative).* Reason:
(a) keeps a like-for-like continuum MC-vs-GHB comparison, whereas (c) drops MC
entirely; 400 or 850 kPa covers the stress range of the middle-to-lower
benches. Options as considered:
- (a) rock-mass MC equivalents: Hoek, Carranza-Torres & Corkum (2002) c, phi
  of each stratum's own GHB envelope over 0..sig3max
  (`strength.ghb_equivalent_mc`, reproduces the Phase-1 stored equivalents
  to 0.03%). `ssr_sweep.py --mc-strength rockmass --mc-sig3max <Pa>`. Needs
  its own sig3max.
- (c) no MC-SSR; the GHB-PINN FOS is compared against the LEM F = 0.94.

*Analysis note on the sig3max value.* D-5.3 fits GHB over 0–179 kPa, so at
400 kPa the MC equivalents linearise the envelope over a wider confinement
range than the GHB reduction uses; the two criteria are then not fitted over
the same range, which qualifies "like-for-like" and must be stated with the
MC-vs-GHB difference. Pointwise on the baseline at SRF 1 the 400 kPa
equivalents leave Mk_d 0.533 admissible (179 kPa-range GHB: 0.389; 850 kPa MC:
0.655). Rock-mass equivalents at 400 kPa (Hoek 2002): Mk c 92.4 kPa phi 27.0;
Mk_d c 33.4 kPa phi 13.5.
`--mc-strength bedding` remains only to reproduce the (d) evidence.

*Analysis note.* Under (a) the MC-vs-GHB comparison is between two fits of the
same rock-mass envelope; it measures linearisation, not a different physical
strength. CPU preview (50 epochs, N_PDE 400, sig3max 179 kPa, `ref`): Mk_d
admissible 0.986 at SRF 1 falling to 0.900 at SRF 1.15, so (a) is workable.

### D-5.3 — GHB fit range: sig3 0 to 179 kPa

*Decision.* `--sig3-lo 0 --sig3-hi 1.79e5` (Pa), set on the marl p99 stress
state to avoid the over-confinement of the Day-39 850 kPa default.

*Analysis note.* Mk_d area-weighted sigma_3 at t = 30 d: p95 130.7 kPa, p99
179.4 kPa (Mk p99 133.2 kPa). 850 kPa lies near Tm's p75 (793 kPa). Pointwise
the range has a small effect: Mk_d admissible at SRF 1.25 is 0.192
(0–131 kPa) vs 0.190 (0–850 kPa); the reduced envelopes agree to ~3 kPa over
Mk_d's stress range and differ most at sigma_3 = 0 (27.5 vs 24.4 kPa). 12.1% of
Mk_d (area) is in tension, below the fit range; tension is governed by the GHB
cutoff, see the `make_fig6` clip note.

### D-5.4 — Disturbance factor D = 1 retained

*Decision.* D = 1 for all strata as stored, accounting for heavy production
blasting and stress relief in the upper benches of the Sekkoy formation.

*Analysis note, consequence for reporting.* D is the largest lever found. GHB
admissibility of Mk_d at SRF 1 on the elastic baseline: 0.389 (D = 1), 0.906
(D = 0.7), 0.989 (D = 0). At D = 1, 61% of Mk_d (about 6% of the area) is
pointwise outside the envelope before reduction, the Day-40 "baseline already
failing" result. Under D-5.1 the SRF 1 fine-tune moves the field toward
admissibility (CPU preview: Mk_d 0.971 under `ref`), so the sweep reference is
a penalty-adjusted field, not the elastic baseline; Methodology must say so.
A D sensitivity arm would need re-solved sigma_0 (D enters E_rm) and is not
scheduled. `freeze_baseline.py --accept-elastic-overstress` exists because
this state trips the Day-34 elastic-check gate by construction.

### D-5.5 — Admissible-drop metric: min_tag primary, tag:Mk_d secondary

*Decision.* `min_tag` is the primary execution value: the default of
`--admissible-metric`, and the only metric `_has_failed` uses. It tracks
whichever stratum reaches critical yield first, so the detector is not blind
to an unexpected failure location. `tag:Mk_d` hardcodes an expectation of
where failure occurs, so it is recorded as a secondary, strata-specific
metric: `--admissible-secondary` (default `tag:Mk_d`), written at every SRF
(`admissible_secondary` in sweep.jsonl) and in result.json with explicit
`admissible_metric_role` / `admissible_secondary_role` fields, never used to
detect failure (`test_has_failed_ignores_the_secondary_metric`).

*Decision: `area` rejected*, masked by Tm's 87% area share.

*Analysis note.* Complete yielding of both marls moves `area` by at most
0.068; `ssr_sweep.reachable_drop` reports this at the reference state.
Pointwise, min_tag and tag:Mk_d coincide while Mk_d is the least admissible
stratum.

*Decision (second ruling): `--admissible-drop 0.15`*, chosen to capture a
clean fall in admissible fraction without firing on small fluctuations.

*Analysis note.* The cited basis, 0.389 at SRF 1 to 0.192 at SRF 1.25, is the
POINTWISE elastic baseline. The sweep's reference is the penalty-adjusted
field after the SRF 1 fine-tune (CPU preview: 0.971 under `ref`, 0.800 under
`raw`), so those numbers are not the trajectory the threshold will see.
Confirm 0.15 against the GPU smoke trajectories before the full sweeps; the
preview fell 0.064 (`ref`) and 0.121 (`raw`) over SRF 1.00–1.15 at 50 epochs.

### D-5.6 — baseline-v1 for validation; production baseline before headline numbers

*Decision.* `baseline-v1` (runs/ansatz/exp_seed7, 2,000 epochs) is used for
code testing and smoke sweeps. A fully converged production baseline must be
trained before any headline FOS is reported, because baseline-v1 shows
numerical creep artefacts that would contaminate published values: the nominally fixed
base (u = v = 0) moves about 1.4 mm by t = 30 d. *Decision (wording):*
"numerical creep artefacts" is the student's term and is retained.

*Enforcement (analysis note on implementation).* `ssr_sweep.py` records the
baseline checkpoint's SHA-256, step and epochs in result.json.
`collect_results.py` marks any run whose baseline hash matches
`baselines/baseline-v1/MANIFEST.json` as `validation_only:baseline-v1` (and
runs with no hash as `unknown`); `make_ssr_figs.py` stamps figures drawn from
such runs VALIDATION ONLY. Identity is by hash, not path or epoch count, so a
copied checkpoint is still caught and no epoch threshold had to be invented.

*Analysis note, found while recording.* `configs/production_labpc.args`
carries no `--ansatz`, so `train.py @configs/production_labpc.args` trains the
`unbounded` ansatz that D-A.1 rejected; its `--out runs/prod01` also points at
an existing run. `scripts/run_production_baseline.bat` requires the ansatz,
writes to `runs/prod_baseline_v2`, refuses an existing log, and freezes as
`baseline-v2`. Wording was raised: in geomechanics "creep" also names
time-dependent material deformation, so define the term at first use in
Methodology as a numerical artefact of the under-converged baseline.

### D-5.7 — Production baseline configuration

*Decision: `--ansatz cap`* (D-A.1 cap vs exp). Reason given: bounded behaviour
for numerical stability over long training.

*Analysis note.* In this code the ansatz modes bound the pore-pressure head
psi (psi <= 0), not activations, output scaling or stress gradients; the
stability argument recorded in D-A.1 is about psi reaching saturation. There
is also no Fault 17 in the Section 5 domain (F1 only). **Seed conflict:** in
the D-A.1 ablation `cap` failed on exactly one seed, 20250812
(`pde_richards` ratio 0.418 against 2.3e-05 and 7.9e-05 on seeds 1234 and 7;
`psi_max` pinned at -1e-40, i.e. on the cap), and
`configs/production_labpc.args` sets `--seed 20250812`. D-A.1 asked for one
more seed per arm before concluding. `run_production_baseline.bat` therefore
requires `SEED` explicitly.

*Decision: `--lbfgs-epochs 500`* (500–1000 named) as a terminal polish after
100k Adam epochs.

*Analysis note.* Each L-BFGS iteration costs up to `--lbfgs-max-eval` (25)
closure evaluations, so 500 iterations can cost up to ~12,500 Adam-epoch
equivalents (~2 h on the 4060 Ti). Weights are frozen at handover; the
`_lbfgs_phase` docstring warns this can drive a degenerate psi field further
into its regime while the loss looks excellent. Check `psi_max_m` and `bc` in
the log after the polish.

### D-5.8 — Reporting choices

*Decision: Fig 9 at t = 30 d* (`make_fig9.py --t 30`). *Analysis note:* the
reason given refers to transient analytical drawdown and inflow curves; the
Nguyen–Raudkivi drawdown validation belongs to Section 7 and has no
counterpart in the Section 5 domain, so the reason recorded is the end of the
30-day rainfall window. The separate choice of WHICH SRF state (`failed` or
`stable` bracket end) is still open; `t` and the state are different settings.

*Decision: reported `w_yield` = 1.0*, the centre of the 0.3/1/3 sweep,
**conditional on the plateau holding**. *Analysis note:* the centre value does
not by itself show FOS independence; the Fig 10 spread across all three does.
If the spread is large, 1.0 is not reportable as-is.

### D-5.13 — Failure criterion CALIBRATED to FOS = 1.5 (supervisor, Day 44)

*Decision.* The SSR failure criterion is calibrated, not measured. Settings,
applied identically to every sweep in Phases 5 and 6:

| parameter | value | flag |
|---|---|---|
| loss plateau factor | 5 | `--plateau-factor 5` |
| admissible criterion | absolute floor | `--admissible-mode floor` |
| admissible floor | 0.50 | `--admissible-floor 0.50` |
| displacement factor | 5 | `--disp-factor 5` |
| metric | min over strata | `--admissible-metric min_tag` |

*Reason given.* The default readout, FOS 3.449 at plateau factor 20, is an
artefact of the soft constraint's gradual yielding. Calibrating to 1.5 grounds
the framework in practical geotechnical design standards.

*Evidence this replaces (`runs/probe_v2_disp5`, baseline-v2, GHB, w_yield 1,
raw).* No threshold-independent FOS exists:

| plateau factor | 5 | 10 | 15 | 20 | 25+ |
|---|---|---|---|---|---|
| FOS | 1.500 | 2.000 | 2.750 | 3.453 | > 4.0 |

| admissible floor | 0.60 | 0.50 | 0.40 | 0.30 | 0.20 | 0.15 |
|---|---|---|---|---|---|---|
| SRF | 1.250 | 1.500 | 1.750 | 2.250 | 3.000 | 3.469 |

*Analysis notes — binding on how the number is reported.*

1. **This is calibration, not prediction.** The thresholds were selected to
   produce 1.5. FOS_GHB is therefore not an independent result of the model,
   and no text may present it as one. What remains predictive is everything
   RELATIVE under the same fixed criterion: MC vs GHB, the sensitivity
   ranking (Fig 12), the coupling effect (Fig 11), the w_yield plateau.
2. **The agreement of the two thresholds at 1.500 is a grid coincidence.**
   Both crossings are bracketed in (1.25, 1.50) on the 0.25 grid: the plateau
   condition needs L > 5 x 0.2003 = 1.0015 (L = 0.600 at 1.25, 1.057 at 1.50)
   and the floor needs adm < 0.50 (0.582 at 1.25, 0.475 at 1.50). Bisected on
   a finer grid the two will generally NOT coincide; expect a value near
   1.3-1.45. Report the bisected number with its bracket, not "1.500".
3. **Uncertainty has three levels**, and only the smallest is the bracket:
   bisection +/- 0.004; run-to-run optimisation noise ~ +/- 0.05 (neighbouring
   SRF states are non-monotone in loss, e.g. 3.9638 at SRF 3.438 vs 3.9548 at
   3.445); criterion choice ~ +/- 1.
4. **The displacement condition is non-informative on baseline-v2.** The
   reference field is 1.12e-6, so any yielding gives ratios of 10^2 to 10^3
   and the condition fires at the first step whatever the factor. It is
   retained for continuity; the criterion is effectively two-condition.
5. **Open tension with the case study.** 1.5 is a design-ACCEPTANCE value,
   while Section 5 is a slope that moved, which is why Ulusay et al.
   back-analysed it to F = 0.94 (< 1). A criterion calibrated to return 1.5
   here should expect the question "why does a failed slope score 1.5?".
   An alternative anchor, not adopted: calibrate the criterion to reproduce
   F ~ 0.94 under the paper's residual strength, then report what GHB gives
   under that same criterion. This keeps the calibration tied to this site
   rather than to a generic standard.

*Implementation.* `ssr_sweep.py` gains `--admissible-mode {drop,floor}` and
`--admissible-floor`; `result.json` records the mode, the floor and
`calibrated: true`. Both batch files carry the settings. Tests:
`test_floor_mode_fires_on_the_absolute_value_not_the_drop`,
`test_has_failed_still_needs_all_three_under_floor_mode`.

## Open issues register (Day 42)

Everything unresolved across Steps 5–6, including items raised in chat and
docs but never recorded here. **Nothing in this section is a decision.** When
one is made, write it as a D-5.x entry and mark the row closed. IDs are
stable; do not renumber. `docs/open_items.md` predates this register and
covers Steps 1–4.

Status: ⛔ blocks GPU work or a figure · ◐ affects reporting · ○ housekeeping

### Detector and sweep settings

**O-1 ⛔ `--yield-norm`: `raw` or `ref`.** Blocks every sweep.
GPU smoke (baseline-v1, N_PDE 10,000): L_yield on the baseline is 0.450, so
`ref` is ~2.2× stronger than `raw` at `w_yield` = 1 (the earlier "38×" came
from a 400-point CPU preview and is withdrawn). 4-step smoke, Mk_d admissible:
`ref` 0.969→0.954, `raw` 0.948→0.929. The `ref` scale depends on N_PDE, so
under `ref` the Phase 6 N_PDE arms each run at a different effective penalty.
Not valid: "1e-3 / 1e-4" (not a mode that exists).

**O-2 ✅ CLOSED by D-5.13 (calibrated floor 0.50).** Original entry:
`--admissible-drop` 0.15, or the drop condition itself. Blocks any
FOS. Full-length probe `probe_ghb_ref` (baseline-v1, GHB 0–179 kPa, `ref`,
`w_yield` 1, 1,500 epochs/SRF): min_tag 0.972, 0.950, 0.897, 0.838, 0.846 at
SRF 1.0–2.0. Loss ratio reached 23.8 (1.75) and 33.2 (2.0); displacement
ratio 10.3 at 2.0; the largest drop, 0.134, never exceeded 0.15, so no failure
was declared. Admissibility stalled near 0.84 as the penalty pulled stress
back. Risk: if the drop condition controls when failure is declared, FOS
could rise with `w_yield` because of the detector, not the physics, which
would defeat the D-5.1 plateau. Options: keep 0.15; lower it; redefine it (for
example, a step-to-step drop or a rate); or let loss plus displacement decide.
Pending: `probe_ghb_raw`.

**O-7 ✅ CLOSED by D-5.13 (plateau 5, disp 5).** Original entry:
`--plateau-factor` 20 and `--disp-factor` 10 are inherited, uncalibrated values. They decide the FOS jointly with O-2. Probe: the
displacement condition was only just met at SRF 2.0 (10.3 against 10).
Record them as decisions or change them, together with O-2.

**O-8 ◐ SRF step and warm-start bias.** Default `--srf-step 0.05` needs ~20
evaluations to reach SRF 2.0 (~6 h per sweep); the probe used 0.25 (~10
evaluations with bisection, ~2.7 h). Larger warm-start jumps are what
`run_phase5.bat cold` measures. Choose the step for production sweeps.

### Production baseline

**O-3 ⛔ Production `SEED`.** `configs/production_labpc.args` sets 20250812,
the one seed on which `cap` failed in D-A.1 (`pde_richards` L/L0 0.418 vs
2.3e-05 and 7.9e-05; psi pinned at the cap). Seeds 7 and 1234 passed under
cap. Blocks `run_production_baseline.bat train`.

**O-6 ◐ Acceptance criterion for the production baseline.** D-5.6 requires a
converged baseline but defines no criterion. `scripts/check_baseline.py` on
baseline-v1: base drift 1.294 mm (u) and 1.399 mm (v) at t = 30 d (0.153 /
0.086 mm at t = 0); least-converged term `bc_mech` L/L0 1.27e-02; others
1.3e-04 to 6.8e-03. Decide what drift and ratios make "converged" before
reading the v2 numbers, not after.

### Figures and reporting

**O-4 ⛔ Fig 9 SRF state: `failed` or `stable`.** Distinct from t = 30 d
(D-5.8). Blocks `run_phase5.bat figs`.

**O-5 ◐ Fig 10 layout.** The workbook defines Fig 10 as FOS bars (LEM /
MC / GHB); D-5.1 makes the FOS-vs-`w_yield` plateau compulsory. Implemented
both: `fig10` = bars with plateau range as error bars, `figS1` = plateau.
Decide: plateau as a Fig 10 panel, supplementary, or renumber.

**O-15 ◐ LEM comparison framing.** Ulusay's F = 0.94 uses residual bedding
strength in limit equilibrium; the PINN uses rock-mass GHB (and rock-mass MC
under D-5.2 a). The baseline-v1 probe declared no failure to SRF 2.0. Decide how
Results and Discussion present a PINN FOS that may sit well above 0.94, before
the number exists.

**O-19 ◐ Fig 1 source.** Cannot be generated from the repo; redraw from
Ulusay et al. (2014) with citation, or obtain permission to reproduce.

### Physics and methods consistency

**O-13 ◐ Yield on total or effective stress.** `loss.L_yield` checks total
stress sigma_0 + D:eps with the Bishop increment; the
`plasticity.yield_violation` docstring describes its inputs as effective
principal stresses. Methodology must state one, and the code must match it.

**O-14 ◐ Tension clip in `make_fig6` FS.** Clipping sigma_3 < 0 to zero
credits tensile points with the rock-mass UCS; 147 Mk_d points within 3 m of
the ground lie beyond the GHB tensile cutoff yet read FS >= 1 (Day 40). Keep
and report, or change the FS definition.

**O-11 ◐ GSI arm and sigma_0.** The GSI arm moves E; sigma_0 is not
re-equilibrated (`sigma0_consistent: false`). Re-solve sigma_0 per arm
(`17_sigma0_solve.py`) or state as a limitation.

### Phase 6 scope

**O-9 ⛔ `KS_STRATUM` and `GSI_STRATUM`.** Replace the plan's coal-seam K_s
arm (no coal in Section 5). Blocks `run_phase6.bat arms`.

**O-10 ◐ Rainfall arm.** 10/20/40 mm/hr give bit-identical boundary conditions
under `INFILTRATION_MODE = "capacity_limited"` (test-pinned). Drop the arm, or
change the infiltration mode (a model change).

**O-12 ◐ Replicates.** `--sampling-seed` varies collocation draws only; true
network-seed replicates need one production baseline per seed (~18 h each).

**O-20 ⛔ The baseline checkpoint was overwritten in place (Day 42).**
`runs/ansatz/exp_seed7/ckpt_final.pt` now holds a different network:
`eps_uv` 1e-3 -> 1e-2, L_yield(baseline, SRF 1) 0.450 -> 22.636, |u| at
SRF 1 1.06e-3 -> 6.88e-3. Consequences: (i) every earlier run on that path
(the smoke runs, both probes, the Day-40/41 evidence) is not comparable with
anything run after it; (ii) `probe_ghb_raw/result.json` was rewritten by a
`--cold-start` command that silently resumed the nine warm records computed on
the OLD checkpoint, so that file's provenance is wrong; (iii) an unfrozen
checkpoint can be replaced without trace. Code changes made in response, not
decisions: `ssr_sweep.py` writes `run_fingerprint.json` and refuses to resume
when the baseline hash, cold-start flag or any sweep setting differs;
`collect_results.py` marks any baseline not in a `baselines/*/MANIFEST.json`
as `unfrozen_baseline`, never `eligible`.
**Open:** whether `eps_uv` = 1e-2 is adopted (a model change: it scales the
network's displacement contribution, and the Step 3.4 balancer weights were
tuned for 1e-3), which checkpoint each future run uses, and whether the
affected probe directories are discarded or kept as pre-change evidence.

**O-21 ◐ The hydraulic problem as posed is essentially static, and the
domain is unsaturated throughout.** Measured on the 100k run
`runs/ansatz_epsuv1e-2` (200x200 grid, 34,185 interior points):

| t (d) | psi median | psi min | psi max | psi >= 0 fraction |
|---|---|---|---|---|
| 0 | -64.37 | -135.72 | -3.13 | 0.0000 |
| 30 | -64.38 | -135.74 | -3.23 | 0.0000 |

largest |psi - psi_0| at t = 30 d: **0.248 m**. The cause is structural, not a
solver failure: `boundaries.Z_WT` = 197.0 m a.s.l. is **3 m below the domain
floor** (`geometry.Z_BASE` = 200 m; `12_bc_plot.py` already labels it "below
domain"), so the initial condition is hydrostatic suction psi = Z_WT - z and
the solution stays there. psi_max = -3.23 m against Z_WT - Z_BASE = -3.0 m;
psi_min = -135.74 m against 197 - 332.7 = -135.7 m at the bench.

Consequences to state in Results and Limitations, not to fix silently:
1. **No saturated zone exists anywhere**, at any time, so Tm is unsaturated
   throughout despite being described as the aquifer.
2. The `interface` term is FROZEN at L = 7.7e-28 with ||grad L|| = 8.4e-31
   because a hydrostatic field satisfies flux continuity across every contact
   identically. Not a dead gradient; a degenerate solution.
3. `pde_richards` at 8.7e-11 is not evidence of a well-resolved transient:
   static hydrostatic suction satisfies Richards trivially.
4. The workbook's Fig 5 caption ("wetting front + drawdown cone") cannot be
   produced by this section: the drawdown cone belongs to Section 7, and the
   rain BC is capacity-limited (D-A.4, O-10).
5. **The HM coupling has little to couple.** Effective stress moves only with
   a 0.248 m suction change over 30 days, so the Fig 11 one-way vs two-way
   difference is likely to be near zero. That is a legitimate result, but it
   is central to the thesis's framing and must be decided deliberately.

Options: accept and report it as the physics of Section 5 under these
boundary conditions; or revisit the water-table datum, which
`docs/open_items.md` still carries as an open Step 2.3 item (the workbook says
"water-table elevation still to be set from the head data", and Z_WT = 197 is
described as the top of the karstic head range). Changing it invalidates
sigma_0, the IC cache and every trained baseline.

### Housekeeping

**O-16 ✅ CLOSED (Day 42).** D-A.1 now carries a pointer to D-5.7 and to the
seed conflict.

**O-17 ◐ Stale text — repo files corrected (Day 42), external documents
still open.** Corrected in place, as dated notes rather than rewrites:
`docs/day39_build_check.md` §3a, `docs/STEP6_STATUS.md` (N_PDE cost and the
coal K_s arm), `boundaries.flux_bc` docstring. **Still to fix outside the
repo:** the 60-day workflow document still lists a coal-seam K_s arm and the
"~21.5%" coupling figure; correct them before they reach the thesis.

**O-18 ✅ CLOSED (Day 42).** The shadowed stub was removed; the live
definition is unchanged, and
`test_loss_bc_mech.py::test_mechanical_bc_is_defined_exactly_once` asserts on
the source so a second definition cannot reappear unnoticed.
