r"""Evidence for restating D-S.4. Are the two Richards terms O(1)?

D-S.4 closed Step 1.6 as `normalise="none"` on this argument:

    Pi_R_diff = 1.4948e-3 and Pi_R_grav = 1.5401e-3; their ratio is exactly
    L_ref/H_ref = 1.030. Capillary diffusion and gravity drainage are balanced
    to 3%, so no term dominates INSIDE the Richards residual and a constant
    divisor has nothing to correct.

Two things are being conflated there. The Pi groups are the COEFFICIENTS. The
terms are

    storage     C*(psi*) dpsi*/dt*
    diffusion   Pi_R_diff div*[K*(psi*) grad* psi*]
    gravity     Pi_R_grav dK*/dz*  =  Pi_R_grav (dK*/dpsi*)(dpsi*/dz*)

so the ratio of the terms is the ratio of the Pi groups multiplied by whatever
K*, dK*/dpsi* and the field's own derivatives do. Under van Genuchten with
n = 1.2 those span six to nine orders across this domain. A 3% agreement
between two coefficients licenses no claim at all about the terms they
multiply.

This script tabulates the ratios that actually decide it, per unit, over the
full psi range the Section 5 IC covers (psi_0 = -(z - 197), so -3 m at the
domain floor to -161 m at the crest):

    D*      = Pi_R_diff K* / C*                 diffusion / storage
    G*      = Pi_R_grav (dK*/dpsi*) / C*        gravity / storage
    D*/G*                                       diffusion / gravity

D* and G* are the terms' ratios with the unit derivative factors stripped out
(dpsi*/dt* = dpsi*/dz* = d2psi*/dz*2 = 1). That is the honest comparison: it
asks what the COEFFICIENT STRUCTURE does to a field whose derivatives are all
O(1), which is the assumption "both terms are O(1)" actually rests on.

    PYTHONPATH=. python scripts/ds4_term_scales.py
    PYTHONPATH=. python scripts/ds4_term_scales.py --json docs/ds4_term_scales.json
"""
import argparse
import json

import numpy as np
import torch

from src.materials import load_materials
from src.nondim import SCALES
from src.residuals import swcc_from_material

PSI_M = (-0.01, -0.1, -1.0, -3.0, -10.0, -30.0, -80.0, -161.0)


def ratios(psi_m, mat, s=SCALES):
    ps = torch.tensor([psi_m / s.H_ref], dtype=torch.float64,
                      requires_grad=True)
    swcc = swcc_from_material(mat, s)
    K = swcc.K_star(ps)
    C = swcc.C_star(ps)
    dK, = torch.autograd.grad(K.sum(), ps)
    D = (s.Pi_R_diff * K / C).item()
    G = (s.Pi_R_grav * dK / C).abs().item()
    return {"K_star": K.item(), "C_star": C.item(), "D_over_S": D,
            "G_over_S": G, "D_over_G": D / G if G > 0 else float("inf")}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    s = SCALES
    mats = load_materials()
    out = {"Pi_R_diff": s.Pi_R_diff, "Pi_R_grav": s.Pi_R_grav,
           "Pi_ratio": s.Pi_R_grav / s.Pi_R_diff, "rows": []}

    print(f"Pi_R_diff = {s.Pi_R_diff:.4e}   Pi_R_grav = {s.Pi_R_grav:.4e}   "
          f"ratio = {s.Pi_R_grav / s.Pi_R_diff:.4f}  (= L_ref/H_ref)\n")
    print("D* = diffusion/storage, G* = gravity/storage, with all field "
          "derivatives set to 1.\n")

    for u in ("Mk", "Mk_d", "Tm"):
        print(f"{u}   (K_s = {mats[u].K_s:.3e} m/s, alpha = {mats[u].alpha}, "
              f"n = {mats[u].n})")
        hdr = (f"  {'psi (m)':>9}{'z (m)':>8}{'K*':>12}{'C*':>12}"
               f"{'D*':>12}{'G*':>12}{'D*/G*':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for p in PSI_M:
            r = ratios(p, mats[u], s)
            r.update(unit=u, psi_m=p, z_m=197.0 - p)
            out["rows"].append(r)
            print(f"  {p:>9.2f}{197.0 - p:>8.0f}{r['K_star']:>12.3e}"
                  f"{r['C_star']:>12.3e}{r['D_over_S']:>12.3e}"
                  f"{r['G_over_S']:>12.3e}{r['D_over_G']:>10.3g}")
        print()

    # Where does each stop being O(1)? Bisect on the ratio = 0.1 crossing.
    # Both ratios increase monotonically toward saturation, so the bracket is
    # [dry, wet] = [-300, -1e-4] and a ratio below 0.1 at the WET end means
    # there is no crossing: the term is never O(1) anywhere.
    print("psi at which each ratio reaches 0.1 (i.e. where the term becomes "
          "O(1) against storage):")
    for u in ("Mk", "Mk_d", "Tm"):
        for key, name in (("D_over_S", "diffusion"), ("G_over_S", "gravity")):
            dry, wet = -300.0, -1e-4
            if ratios(wet, mats[u], s)[key] < 0.1:
                print(f"  {u:>5} {name:<10} never reaches 0.1 -- at psi = "
                      f"{wet:g} m it is still "
                      f"{ratios(wet, mats[u], s)[key]:.2e}. This term is "
                      f"O(1) NOWHERE, at any saturation.")
                out.setdefault("psi_crit", {})[f"{u}_{name}"] = None
                continue
            for _ in range(80):
                mid = 0.5 * (dry + wet)
                if ratios(mid, mats[u], s)[key] < 0.1:
                    dry = mid         # crossing is wetter than mid
                else:
                    wet = mid
            psi_c = 0.5 * (dry + wet)
            z_c = 197.0 - psi_c
            where = ("BELOW the domain floor -- never reached in Section 5"
                     if z_c < 200.0 else f"z = {z_c:.2f} m a.s.l.")
            print(f"  {u:>5} {name:<10} psi = {psi_c:>9.4f} m  -> {where}")
            out.setdefault("psi_crit", {})[f"{u}_{name}"] = psi_c
        print()

    print("Read against the domain: the floor is z = 200 m (psi_0 = -3 m) and "
          "the crest is\nz = 358 m (psi_0 = -161 m). A critical psi drier "
          "than -3 m means the term is\nO(1) somewhere in the domain; a "
          "critical psi wetter than -3 m means it is\nO(1) NOWHERE in Section "
          "5 on the initial condition.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())