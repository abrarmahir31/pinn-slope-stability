# PINN slope stability — Step 3.4 handoff

Paste this at the start of a new chat to continue.

---

## Project

Physics-informed neural network for coupled hydro-mechanical slope stability,
Section 5 of the Isikdere open-pit lignite mine (Ulusay et al. 2014). Repo
`~/thesis-pinn`, branch `step-3.2-loss`, remote
`github.com/abrarmahir31/pinn-slope-stability`. **Python 3.10** (conda env
`pinn`, miniforge), WSL Ubuntu. Test suite: **185 passing, 4 skipped**.

Three strata: **Mk** (Sekköy marl, aquiclude), **Mk_d** (weak zone in Sekköy,
aquiclude), **Tm** (crystallised limestone basement, karstic).

**Step 3.2 is COMPLETE on the hydraulic side.** Mechanics is Step 3.4 by
coupling order — see "Why 3.2 closed without mechanics" below.

---

## Repo conventions — read before writing code

**Layer separation.** `nondim.py` holds residual algebra (no autograd).
`residuals.py` is the autograd layer. `properties.py` is the single source of
truth for parameters, reading `Phase 1/isikdere_phase1_dataset.json`.
`materials.py` is a thin adapter over `properties.py` + `vg.py`. Never
reimplement the constitutive model.

**Imports are `src.`-prefixed**: `from src.residuals import ...`. There is no
packaging (no pyproject/setup.py); `tests/conftest.py` puts `src/` and
`src/step21_geometry/` on `sys.path`. New imports go in the existing import
block, never at line 1 — `from __future__ import annotations` must come first.

**`fields` is a callable, not a PINN.** Signature `fields(x, z, t) -> psi*`,
shape (N,1). `loss._psi_field(net)` wraps a network into one. Loss functions
take `net` and wrap internally, so an analytic test field can be passed
directly where a network is expected.

**Squared residuals accumulate across sets, divided by the GLOBAL weight sum.**
Per-set parts are per-set MEANS. These are different normalisations and the
parts do NOT sum to the total — see DECISIONS.md D-3.2.2. `total_loss` adds
`contrib_*` entries that DO sum.

**`Collocation`** (frozen dataclass, `sampling.py`): `x, z, t` (leaf tensors
(N,1), requires_grad), `w` (N,1), `tag` (np array of str), `nx`, `nz`
(optional (N,1) outward normals, boundary and interface sets only).

**Geometry queries happen ONCE, at sample time, and travel with the
Collocation.** The round-trip x → x/L_ref → x moves boundary points by ~1e-16,
enough for `material_tag` to return `"outside"` on curved segments.

**Every claim gets a test, including negative controls.** Assert the thing is
zero when it should be AND nonzero when it should not be, or the first
assertion is satisfied by `return 0.0`.

**Every `!= pytest.approx(...)` needs an explicit `abs=`.** Quantities here
routinely live at 1e-12 and below, where the default `abs=1e-12` makes a
negative control silently vacuous. Use `abs=1e-40`.

---

## Call signatures — check these before writing a call

```
L_PDE(net, coll, mats, s=SCALES, per_tag=False)
L_IC(net, coll, psi0_star)                       # 3 args, no mats, no s
L_BC(net, bcs, mats, s=SCALES, per_segment=False)
L_interface(net, ifaces, mats, s=SCALES, per_contact=False)
total_loss(net, coll, bcs, ic, ifaces, mats, s=SCALES,
           weights=None, include_mechanics=False, per_term=False)

richards_residual(fields, x, z, t, mat, s=SCALES,
                  normalise="none", return_terms=False)
richards_residual_expanded(fields, x, z, t, mat, s=SCALES)
darcy_flux(fields, x, z, t, mat, s=SCALES)              -> (qx, qz)
interface_flux_jump(fields, x, z, t, mat_a, mat_b, nx, nz, s=SCALES)

sample_interior(n, seed)                    -> Collocation
sample_boundary(n=2000, seed=0, t_max=T_MAX_DAYS)  -> dict[str, Collocation]
sample_interfaces(n=500, seed=0, t_max=T_MAX_DAYS) -> dict[str, Collocation]
load_initial(path)                          -> (Collocation, targets)
ic_targets(path=None, s=SCALES)             -> (Collocation, psi0_star)
flux_bc(segment, pts, tag=None)             # pts is PHYSICAL (N,2), m
materials.C_star(psi, mat, s=None)          # psi in METRES, returns C*
materials.load_materials()                  -> dict[tag, Material]
```

**Parts-dict key prefixes:** `bc_<segment>`, `interface_<contact>`,
`contrib_<term>`, plus bare `bc`, `interface`, `total`, `pde_richards`,
`ic_head`, `ic_disp`.

`darcy_flux` and `richards_residual` take ONE material. Boundary and interface
sets are mixed-tag, so partition by `coll.tag` and call per material. Rebuild
each slice as a fresh leaf (`coll.x[idx].detach().clone().requires_grad_(True)`)
rather than indexing a view.

---

## Scales (`nondim.py`)

```
L_ref = 170.0 m      T_ref = 86400 s      dtheta_ref = 0.33
H_ref = 165.0 m      K_ref = 1e-6 m/s     sig_ref = 1e6 Pa
Pi_R_diff = T_ref*K_ref*H_ref/(L_ref^2*dtheta_ref)
Pi_R_grav = K_ref*T_ref/(L_ref*dtheta_ref) = 1.5401e-3
Pi_R_hz   = H_ref/L_ref = 0.9706
Q_ref     = L_ref*dtheta_ref/T_ref = 6.4931e-4 m/s     [flux scale]
E_ref     = 1.0e8 Pa   ->  U_ref = sig_ref*L_ref/E_ref = 1.70 m
```

`E_ref` was 1e9 until Step 3.2 closed; corrected to 1e8, a marl scale with Tm
(4.26e9, rigid basement) excluded. See DECISIONS.md D-S.1. **The Step 3.2
handoff's `E_ref = 2.0e8` was wrong** — that is `E_rm` for Mk, not a reference
scale.

Identity `Pi_R_diff / Pi_R_grav == Pi_R_hz` is asserted in
`tests/test_interface.py::test_pi_group_identity`.

`tests/test_scale_consistency.py` binds `nondim.py` to the Phase-1 dataset.
Any difference must be listed in `KNOWN_DIVERGENCES` with a reason.

---

## Geometry

```
X_MIN, X_MAX = 0.0, 268.0      Z_BASE = 200.0      Z_WT = 197.0
z_ground(x)      digitised ground surface
z_tm_top(x)      top-of-Tm contact; tiles d_mk_tm (x 0-90) + d_mkd_base (x 91-266)
x_f1(z)          F1 fault, near-vertical (81.4 deg), fitted x = f(z)
X_MK_DIVIDE = 90.7   vertical Mk|Mk_d contact — MODELLING CHOICE, not measured
material_tag(x,z):  below z_tm_top -> Tm; above -> Mk if x <= X_MK_DIVIDE else Mk_d;
                    outside domain -> "outside"
```

Six boundary segments: `natural_ground`, `bench`, `cut_face`, `pit_floor`,
`base`, `far_field_f1`. Segment normals: `pit_floor` is horizontal (-x),
`base` is vertical (-z), `far_field_f1` is +x. A test field with a gradient in
only one direction is INVISIBLE to a segment whose normal is perpendicular to
it — see `test_segment_normals_point_where_the_geometry_says`.

Two interface contacts: `mk_tm`, `mkd_tm`, sampled along `z_tm_top`.

---

## Done in Step 3.2

- `derivatives.py`, `sampling.py`, `L_IC`, `L_PDE` (tag-partitioned)
- Richards residual: 1.1e-17 hydrostatic, mutation-tested, expanded-form cross-check
- `C_star` scale-propagation bug fixed (`ebf8f5c`); `normalise` flag settled
- Interface flux physics (`8330940`): `darcy_flux_nd`, `darcy_flux`,
  `interface_flux_jump`, `tests/test_interface.py`
- E_rm conflict resolved; infiltration capacity cap (`flux_bc` caps at K_s)
- `L_BC` (`ae36172`): six segments, flux + Dirichlet branches
- **`tests/test_loss_bc.py`** (`5b4b017`): 22 tests. Sign convention
  `q*.n = -q_prescribed*` pinned three ways — exact assembly, flipped-sign
  negative control, and the physical ordering that infiltration lowers the
  residual. The negative control is calibrated so peak |q*.n| = 0.5 *
  q_prescribed pointwise; at generic field strength the two conventions differ
  by 1e-3 relative and the test is vacuous.
- **`sample_interfaces`** (`07fccc8`): 19 tests. Arc-length sampling along
  `z_tm_top`, materials identified at +/- 0.10 m, normal from the
  differentiated trace pointing out of Tm.
- **`L_interface`** (`25f0c26`): 10 tests.
- **`total_loss`** (`26f49a5`): 20 tests. `contrib_*` sums to the total;
  `include_mechanics=True` raises.
- **`E_ref` 1e9 -> 1e8** and `tests/test_scale_consistency.py`: 9 tests binding
  `nondim.py` to the Phase-1 dataset.
- **`tests/test_storage_transient.py`**: 8 tests, 30 cases. Closes the C* gap —
  see below.

---

## The C* construction, worth reusing

C* had never run through the residual path: every test field was hydrostatic,
so `dpsi*/dt* = 0` multiplied it by zero.

Take `psi* = f(z*) + k*t*` and evaluate at **t* = 0**. The psi field is then
identical for every k — same suction, same K, same gradient, same `div* q*` —
and only `dpsi*/dt* = k` differs. So

```
R(k) - R(0) == C*(psi*) * k        exactly, pointwise
```

with no cancellation error and without computing `div* q*` at all. Implied C*
matches `materials.C_star` to rtol 1e-9 for all three strata.

Two tests in that file generalise beyond C*:
- `test_storage_term_is_evaluated_at_the_sample_time` samples t* = pi/2, where
  a sin field's derivative vanishes. Every other test in the repo samples
  t* = 0, so a time derivative frozen at zero would pass all of them.
- `test_C_star_respects_the_scales_argument` doubles `dtheta_ref` and asserts
  the storage term CHANGES, without asserting a formula. That is the shape of
  the `ebf8f5c` bug class: an argument that never arrives.

---

## Why 3.2 closed without mechanics

`mechanical_residual` needs an equilibrated `sigma_0` from the gravity warm-up,
which IS Step 3.4. Step 3.2 structurally cannot close before 3.4 opens. Say
this to the supervisor explicitly so it reads as coupling order, not slippage.

Mechanical BCs go with it: `base` fixed, `far_field_f1` / `pit_floor` roller
are cheap Dirichlet conditions, but traction-free on the three exposed segments
needs the stress tensor. Shipping half would make `total_loss` look complete
while omitting the free-surface condition on the cut face — the boundary the
failure mechanism runs through.

`L_IC` already carries a `u*, v*` term pinning displacements to zero at t = 0.
"Mechanics deferred" means the equilibrium residual, not every displacement
term.

---

## Remaining paperwork (no code)

1. **Phase-1 dataset**: correct the `E_ref` justification string — it says the
   moduli "span 8e6 to 2.09e8", which omits Tm at 4.26e9 and names a lower
   bound no stratum has. Update `H_ref_m` to 165 and the `groups` block
   (`Pi_R_hz` there is 0.176 against a live 0.971). Then delete the `H_ref_m`
   entry from `KNOWN_DIVERGENCES` in `tests/test_scale_consistency.py`;
   `test_known_divergences_are_still_divergent` will fail until you do.
2. **`write_step22.sh`**: both `scripts/write_step22.sh` and
   `src/step21_geometry/write_step22.sh` contain `cat > boundaries.py << 'PYEOF'`
   heredocs holding an OLDER version. Running either silently reverts the flux
   limiter and mechanical BC data. Neuter or delete.

---

## Findings that belong in the thesis, not just the code

**1. The rainfall BC delivers essentially nothing.**
Rain 5.56e-6 m/s (20 mm/hr). The entire rainfall surface (`natural_ground`,
`bench`) is at K_s = 1e-9 m/s = 0.004 mm/hr. `16_infiltration_check.py` gives
**TOTAL admitted = 0.02%**. The capped target is 1.54e-6 against network fluxes
of ~1e-3 — numerically identical to no-flow. Per-segment `L_BC` confirms it:
`far_field_f1` dominates at 2.53 against 1.9e-11 for `natural_ground`.

Confirmed independently in `test_prescribed_flux_is_capped_at_saturated_conductivity`:
the cap binds on >99% of rainfall points, which makes `q_prescribed*` uniform
along each segment (both give exactly (K_s/Q_ref)^2 = 2.3719e-12 under the
anchor field).

Three readings for the supervisor: (a) Mk_d's K_s is wrong; (b) the design
storm is wrong (a multi-day low-intensity event infiltrates far more —
`T_MAX_DAYS = 30` hints at this); (c) it is the real result, since
`geometry.py` labels both marls aquicludes from Ulusay's Fig. 19.

**Finding 1 has hardened.** The cut face cannot move water at t = 0 either
(D-3.2.4), so the ground surface AND the cut face are both closed. If (c), the
thesis question becomes how water reaches the failure surface at all, pointing
at the Tm contact and the far-field condition.

**2. Mk and Mk_d are hydraulically identical.**
Same K_s = 1e-9; the weak zone was given the marl's alpha and n by analogy
(Phase-1 `vg_basis: "marl analog, identical alpha/n"`). Tier-B, not measured.
They differ only mechanically (E 3.81e7 vs 2.09e8 Pa). A sheared D=1 zone would
normally be MORE permeable. Top sensitivity candidate — finding 1 rests
entirely on this borrowed number, and `test_prescribed_flux_is_capped...`
will fail the moment it changes, which is correct behaviour.

**3. Tm's effective contrast reverses with wetness.**
K_s(Tm) = 3.13e-6 is 3000x the marls', but at psi* = -0.5 its K_star is 30x
LOWER — coarse-pored material desaturating fast. So the sense of the flux jump
at the contact flips depending on saturation. Real result for the methods
section.

**4. `X_MK_DIVIDE = 90.7` vs trace tiling at 90/91.**
A ~0.7 m strip where the trace came from `d_mkd_base` but `material_tag` says
Mk. `test_material_above_and_below_are_as_the_key_says` passes at 100% of
sampled points, so this is immaterial at present — but it is guarded now, not
merely assumed.

**5. `far_field_f1` normal is approximate.**
`_segment_normal` returns +x; F1 actually dips at 81.4 deg, so it is ~8.6 deg
off. Harmless while the segment stays Dirichlet —
`test_dirichlet_segments_ignore_the_normals` is what makes that true, and
`test_segment_normals_point_where_the_geometry_says` records the orientation.

**6. Tm parameters are Tier C.**
`properties.py` TM block: `vg_theta_r`, `vg_alpha_1pm`, `vg_n` are PLACEHOLDERS
pending citation. Gs = 2.71 is a choice. A 2.2%-porosity rock at K = 3e-6 is
fracture-dominated; any matrix SWCC describes the pathway water does not use.

**7. Tm's modulus is 4.26e9 Pa**, two orders above the marls. Surfaced while
choosing `E_ref`. The Phase-1 dataset's moduli range note omits it entirely.

---

## Practical gotchas

- **Never paste a code block into an existing file.** Use
  `cat block.py >> src/target.py`, which appends at end-of-file regardless of
  cursor position. A 50-line paste landed inside `L_BC` during Step 3.2 and
  produced 25 failures that looked like a `masked_scatter` shape bug. `cat >>`
  is not idempotent — check `grep -c "def <name>"` prints 1.
- **`git checkout HEAD -- <file>` discards import edits too.** After restoring a
  file, re-apply any import lines that were added since the commit.
- **Check the file is SAVED** before running.
- **Boolean masks flatten (N,1) to (K,).** Use
  `idx = torch.as_tensor(np.flatnonzero(coll.tag == tag), dtype=torch.long)`
  and `R.index_copy(0, idx, ...)`.
- **`float(tensor)` on a graph tensor warns.** Use `float(t.detach())`.
- **`residuals._psi_of` returns a bare Tensor unchanged**, so handing it the raw
  net treats u*, v* as extra psi columns. `loss.py` defines its own `psi_of` /
  `uv_of` slicing helpers for this reason.