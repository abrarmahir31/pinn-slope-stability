# Step 3.3 — what to fix, in order

Revised after reading the real `step-3.2-loss` branch. Three items from my
first draft were wrong and are struck out below with what replaced them.

Verified state after adding the three new files: **212 passed, 4 skipped,
1 xfailed** (was 185 + 4).

---

## 1. Sign convention collision in `nondim.py` — DO THIS FIRST

`nondim.py` contains two functions assuming opposite stress conventions.

```python
# line 198 — COMPRESSION positive
def effective_stress_nd(sigma_star, psi_star, chi, s):
    return sigma_star - s.Pi_M_couple * chi * psi_star

# line 180 — TENSION positive (e_z=(0,-1) gives div sigma - Pi_M_body*rho)
def mechanical_residual_nd(div_sigma_eff_star, rho_ratio, e_z=(0.0, -1.0), s):
```

Verified on your branch: two definitions, two docstring mentions
(`residuals.py:7`, `docs/open_items.md:39`), **zero call sites and zero
tests**. The convention is unpinned. Take tension-positive — `sigma = D:eps`
from displacement gradients is tension-positive by construction, and the
alternative means a minus inside the constitutive law that gets forgotten
exactly once.

```python
def effective_stress_nd(sigma_star, psi_star, chi, s: Scales = SCALES):
    """Dimensionless Bishop effective stress. TENSION POSITIVE.

        sigma*_eff = sigma* + Pi_M_couple * chi * psi*

    Positive psi* (pore pressure) unloads the skeleton, so sigma_eff moves
    toward tension. Suction (psi* < 0) with chi = Se adds apparent cohesion,
    moving it toward compression.
    """
    return sigma_star + s.Pi_M_couple * chi * psi_star
```

What that changes numerically, at chi = 1:

| psi* | current | after Fix 1 |
|---|---|---|
| +0.5 | −0.80932 | **+0.80932** |
| −0.5 | +0.80932 | **−0.80932** |

`test_bishop_sign_is_tension_positive` is marked `xfail` and flips to `xpass`
when you apply this. Remove the marker then — an xfail left in place is a test
that has stopped testing.

---

## 2. ~~`SWCC` has no slot for `Theta`~~ — NOT NEEDED

I was wrong. `materials.Se` already exists, is torch-differentiable through
`vg.py`, and its docstring has said *"also the Bishop chi in Step 3.3"* since
Step 1. `materials.theta` likewise. `mechanics.chi_of` / `mechanics.theta_of`
call them with `psi_star * H_ref`, the same single-line unit convention
`swcc_from_material` already uses. **No change to `SWCC` required.** Step 1
left this door open on purpose.

---

## 3. ~~`boundaries.E_RM` disagrees with `properties.E`~~ — ALREADY RESOLVED

Also wrong; I read the stale `main`. On `step-3.2-loss` the two agree exactly
for all three strata (Mk 2.09e8, Mk_d 3.806e7, Tm 4.264e9) and the Poisson
ratios match too. Your handoff's "E_rm conflict resolved" is genuinely done.

Replaced by `test_boundaries_and_properties_agree_on_the_elastic_constants`,
which guards against them drifting apart again — the mechanical residual reads
`properties.py` while the mechanical BCs read `boundaries.py`, and a future
divergence would build those two on different stiffnesses with nothing
visible in any field plot.

**What does still need deciding: Kozeny–Carman on Tm.** From the regenerated
IC cache:

```
area fractions   Mk: 3.1%   Mk_d: 9.7%   Tm: 87.3%
```

Tm is 87% of the domain, `n0 = 0.0221`, and Finding 6 calls it
fracture-dominated. At that porosity a 0.005 strain is a 23% porosity change
against 1.2% for Mk — the KC factor is an order of magnitude more
strain-sensitive there than anywhere else, on the stratum where a
matrix-porosity law has the weakest claim. Switch KC on globally and the
dominant signal in your two-way coupling comes from the material the model
describes least well.

`KCConfig.per_unit` exists for this: `KCConfig(enabled=True,
per_unit={"Tm": False})`. Decide it, write it in DECISIONS.md, and report the
Step 6.2 ablation per-stratum as well as globally — "one-way vs two-way"
measured across the whole domain is mostly a measurement of Tm.

---

## 4. `psi_of` / `uv_of` are now defined twice

`loss.py` has them; `mechanics.py` now has its own, because importing from
`loss.py` would invert the dependency (`loss.py` will shortly import
`mechanics.py`). Note the signatures differ: `loss.uv_of` returns a single
`(N,2)` tensor, `mechanics.uv_of` returns the `(u, v)` pair, because every
caller here unpacks immediately.

Move both to `src/fieldslice.py`, import from there in both modules, delete
the duplicates. Small, but it is exactly the kind of duplication the repo's
layer rule exists to prevent.

---

## 5. `sigma0` has to travel with the `Collocation`

Geometry queries happen once, at sample time, and travel with the
`Collocation`, because the `x -> x/L_ref -> x` round trip moves boundary
points by ~1e-16 and `material_tag` then returns `"outside"` on curved
segments. `sigma0` is a geometry query.

`Collocation` is frozen, so either add a field (updating every construction
site) or carry a parallel `dict[tag, sigma0]`. Adding the field is more churn
now and less later.

---

## 6. Equilibrate `sigma0`, and write down the number

`mechanical_residual` defaults to the collapsed form, which assumes
`div sigma0 + Pi_M_body*rho0_ratio*e_z = 0` pointwise. The raw K0 column does
not satisfy that — the handoff records a 0.20–0.26 `rho*g` horizontal
imbalance on the 31.06° face.

Before using the default: relax `sigma0`, measure the residual imbalance, put
the accepted value in DECISIONS.md as a number. If it will not go small, pass
`div_sigma0_star` explicitly, which needs `sigma0` differentiable in space —
a small pretrained `sigma0` net or an interpolant with an analytic derivative.

Skipping this does not crash. It trains, the loss falls, and it converges to
an equilibrium in which a ~25% body-force imbalance is physics.

`ic_cache.npz` regenerates cleanly and deterministically — I ran
`14_ic_checks.py` here and got all thirteen checks passing, geometry hash
`29ee247510aa`. Confirm that hash matches yours; if it does not, your
`geometry.py` has moved since the cache you have been testing against.

---

## 7. `total_loss` and the config flag

`total_loss(..., include_mechanics=False)` currently raises when True. Wire
it, and add `include_feedback` alongside. Three tests:

1. `include_feedback=False` reproduces the one-way numbers exactly;
2. `include_feedback=True` changes the residual;
3. with `eps_v` forced to zero, True equals False.

Test 3 catches a flag that arrives and is never read. `coupling.py` already
supports 1 and 3 by returning `torch.ones_like` with no graph when disabled —
`test_disabled_factor_carries_no_graph` asserts the graph, not just the value.

Keep the `contrib_*` convention: mechanics adds `pde_mech_x`, `pde_mech_z` as
per-set means, plus `contrib_pde_mech_*` entries that sum into the total.

---

## 8. Settle the step numbering

`residuals.mechanical_residual` says *"blocked on the gravity warm-up (Step
3.4)"* while `PINN_Thesis_60_15_Day_Workflow.docx` makes 3.4 the GradNorm /
optimiser step and 3.3 the two-way coupling. Two schemes, same label.

Take the 60/15-day numbering, since that is the schedule you are measured
against, and re-label the gravity warm-up **3.3a** — it is the first half of
two-way coupling, not a separate step. Write it in DECISIONS.md and use it in
supervisor emails.

---

## 9. Update the handoff

- Delete the "Remaining paperwork" section; all three items verified done
  (`KNOWN_DIVERGENCES` is `{}`, `write_step22.sh.disabled` exists, D-S.4
  committed in `5035b5e`).
- Add the Π_M block to Scales:
  `Pi_M_body = 3.0019`, `Pi_M_couple = 1.6186`, `U_ref = 1.70 m`. Verified
  against the live `Scales`.
- **Correct the attribution.** `Pi_M_couple = rho_w*g*H_ref/sig_ref` contains
  no E. The 0.294 → 1.6186 move came from **H_ref 30 → 165 m**, not from the
  E_ref correction — 1000·9.81·30/1e6 = 0.294 exactly, and the stale
  `Pi_R_hz` of 0.176 × 170 m = 29.9 m. The E_ref correction is what moved
  **U_ref** 0.17 → 1.70 m. Both are in `164182d`, which is why they blurred.
- New gotcha: `eps_v*` is not physical strain; the factor is
  `U_ref/L_ref = 0.01`.

---

## 10. One leftover string

`src/nondim.py` line 61:

```python
E_ref: float = 1.0e8   # Pa  representative rock-mass modulus
                       # (Phase-1 revision; adopted moduli span 8e6-2.09e8)
```

That is the justification string your handoff flagged as wrong — it omits Tm
at 4.264e9 and names a lower bound no stratum has (the lowest is Mk_d at
3.806e7). You fixed it in the Phase-1 dataset in `baf1c17`; this copy still
carries the old claim. Same sentence, two homes, one updated.

Suggested: *"marl-scale reference; Tm (4.264e9) excluded as rigid basement.
Adopted marl moduli 3.81e7–2.09e8. See DECISIONS.md D-S.1."*

---

## Install and verify

```bash
cd ~/thesis-pinn
# paste the three files, then:
PYTHONPATH=. python -m pytest tests/ -q
```

Expect **212 passed, 4 skipped, 1 xfailed**. The xfail is
`test_bishop_sign_is_tension_positive`, waiting on Fix 1.

Line counts: `coupling.py` 177, `mechanics.py` 291, `test_coupling.py` 329.