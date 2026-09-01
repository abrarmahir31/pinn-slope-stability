"""Does Kozeny-Carman feedback have range, and where is it disabled?

Day 24 A/B (fb_on vs fb_off, 200 ep, 2x64) differed by ~3e-13 relative --
consistent with correct wiring acting on ~zero strain, and equally
consistent with a dead mechanism. This distinguishes them by imposing
strain directly rather than waiting for training to produce it.

Second block documents the per-unit exclusion (KC_DEFAULT carries
per_unit={"Tm": False}). Rationale: D-3.3.2 -- Kozeny-Carman is a
matrix-porosity law and Tm is fracture-dominated (n0 = 0.022 against
K_s = 3.13e-06, three orders above Mk). At that porosity the same strain
is a ~23% porosity change against ~1.2% in Mk, so applying a matrix law
to Tm would give the largest spurious response in the domain.
"""
import torch
from src.coupling import KC_DEFAULT, KC_OFF, kozeny_carman_factor
from src.materials import load_materials

mats = load_materials()

print(f"{'tag':>5} {'n0':>8}  {'eps_v':>8}  {'KC_DEFAULT':>16}  {'KC_OFF':>16}")
for tag in ["Mk", "Mk_d", "Tm"]:
    n0 = mats[tag].n0
    for eps_v in [0.0, 1e-4, 1e-3, 1e-2]:
        e = torch.tensor([eps_v], dtype=torch.float64)
        on  = float(kozeny_carman_factor(n0, e, KC_DEFAULT, tag=tag))
        off = float(kozeny_carman_factor(n0, e, KC_OFF, tag=tag))
        print(f"{tag:>5} {n0:8.4f}  {eps_v:8.0e}  {on:16.12f}  {off:16.12f}")
    print()

e = torch.tensor([1e-2], dtype=torch.float64)
f = float(kozeny_carman_factor(mats["Mk"].n0, e, KC_DEFAULT, tag="XX"))
print(f"unknown tag 'XX' -> {f:.12f}  ({'fails OPEN' if f != 1.0 else 'fails closed'})")