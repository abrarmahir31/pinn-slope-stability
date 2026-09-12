r"""How much of the collocation budget lands where Richards carries no signal?

Day 26 evening established that `inspect_richards_terms.py` returns
`cancel = 1` in every row: |R| IS the storage term, and both flux terms sit
8-9 orders below it. That was measured on `runs/prod02/ckpt_002000.pt`, so it
could in principle have been the trained field's doing. This script removes the
network from the question entirely.

It evaluates, on the ANALYTIC hydrostatic IC psi_0(z) = -(z - Z_WT) only:

    D*(z)    = Pi_R_diff * K*(psi_0) / C*(psi_0)       dimensionless diffusivity
    ell(z)   = L_ref * sqrt(D* * t*)                   diffusion length, metres
    c*(z)    = Pi_R_grav * (dK*/dpsi*) / C*            dimensionless celerity
    dz(z)    = L_ref * c* * t*                         gravity drainage, metres

over the t* = 30 day window. A point is INERT when neither mechanism can move
information more than `--reach` metres in the whole window: nothing that
happens anywhere else in the domain can reach it, and nothing it does can
reach anywhere else. The Richards residual there reduces to
C* dpsi*/dt* = 0, which any frozen field satisfies exactly.

This is a property of the parameters and the IC, not of training. It is the
mechanical explanation for `w^[pde_richards]` moving 1264x across collocation
draws (Day 26): the estimator's variance is carried by whichever few points
land in the live band.

NOTE ON GEOMETRY. Z_WT = 197 m and Z_BASE = 200 m, so the water table is 3 m
BELOW the domain floor and is not in the domain at all. "Concentrate points
near the water table" therefore means the bottom few metres of the domain,
which is its closest approach. There is no saturated region to resolve.

    PYTHONPATH=. python scripts/inert_fraction.py
    PYTHONPATH=. python scripts/inert_fraction.py --reach 0.1 --json out.json
"""
import argparse
import json

import numpy as np
import torch

from src.materials import load_materials
from src.nondim import SCALES
from src.residuals import swcc_from_material
from src.sampling import sample_interior

try:
    import geometry as g
except ModuleNotFoundError:
    from src.step21_geometry import geometry as g

Z_WT = 197.0          # vg.psi_initial default; upper bound of the karstic range


def reach_metres(z_phys, mat, s=SCALES, t_days=30.0):
    """(ell_diff, dz_grav) in metres at elevation z on the analytic IC.

    Both are upper bounds on how far information travels in t_days, computed
    from the LOCAL coefficients. A local estimate is the right one here: the
    coefficients vary by six orders over the domain, so a global average would
    be meaningless, and the question is per-point ("is THIS point live?").
    """
    z = torch.as_tensor(np.asarray(z_phys, float), dtype=torch.float64)
    psi_star = (-(z - Z_WT) / s.H_ref).requires_grad_(True)

    swcc = swcc_from_material(mat, s)
    K = swcc.K_star(psi_star)
    C = swcc.C_star(psi_star)
    dK, = torch.autograd.grad(K.sum(), psi_star, create_graph=False)

    D_star = (s.Pi_R_diff * K / C).detach().numpy()
    c_star = (s.Pi_R_grav * dK / C).detach().abs().numpy()

    ell = s.L_ref * np.sqrt(np.maximum(D_star, 0.0) * t_days)
    dz = s.L_ref * c_star * t_days
    return ell, dz, D_star, K.detach().numpy(), C.detach().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reach", type=float, default=1.0,
                    help="metres of travel in 30 d below which a point is "
                         "called inert (default 1.0)")
    ap.add_argument("--n", type=int, default=10000,
                    help="collocation points to classify (production N)")
    ap.add_argument("--sampling-seed", type=int, default=20260808)
    ap.add_argument("--t-days", type=float, default=30.0)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    mats = load_materials()
    s = SCALES
    out = {"reach_m": a.reach, "t_days": a.t_days, "n": a.n,
           "Z_WT": Z_WT, "Z_BASE": g.Z_BASE, "profile": [], "strata": {}}

    print(f"Pi_R_diff = {s.Pi_R_diff:.4e}   Pi_R_grav = {s.Pi_R_grav:.4e}   "
          f"L_ref = {s.L_ref} m   window = {a.t_days:g} d")
    print(f"Z_WT = {Z_WT} m,  domain floor Z_BASE = {g.Z_BASE} m  "
          f"-> the water table is {g.Z_BASE - Z_WT:.0f} m BELOW the domain.\n")

    # ---- 1. profile on the analytic IC -----------------------------------
    zs = np.array([200.0, 202.0, 205.0, 210.0, 220.0, 240.0, 280.0, 358.0])
    hdr = (f"{'z (m)':>7}{'psi0 (m)':>10}" +
           "".join(f"{u + ' K*':>12}{u + ' ell':>10}{u + ' dz':>10}"
                   for u in ("Mk", "Tm")))
    print(hdr)
    print("-" * len(hdr))
    for z in zs:
        row = {"z": float(z), "psi0": float(-(z - Z_WT))}
        line = f"{z:>7.0f}{-(z - Z_WT):>10.1f}"
        for u in ("Mk", "Tm"):
            ell, dz, D, K, C = reach_metres(z, mats[u], s, a.t_days)
            row[u] = {"K_star": float(K), "C_star": float(C),
                      "D_star": float(D), "ell_m": float(ell),
                      "dz_m": float(dz)}
            line += f"{float(K):>12.3e}{float(ell):>10.2e}{float(dz):>10.2e}"
        print(line)
        out["profile"].append(row)

    print(f"\nell = diffusion length over {a.t_days:g} d (m);  "
          f"dz = gravity drainage over the same (m).")

    # ---- 2. classify an actual production draw ---------------------------
    coll = sample_interior(a.n, a.sampling_seed)
    z_phys = (coll.z.detach().reshape(-1).numpy() * s.L_ref)
    live = np.zeros(len(z_phys), dtype=bool)
    per = {}
    for u in sorted(set(coll.tag.tolist())):
        m = coll.tag == u
        ell, dz, D, K, C = reach_metres(z_phys[m], mats[u], s, a.t_days)
        lv = np.maximum(ell, dz) >= a.reach
        live[m] = lv
        per[u] = {"n": int(m.sum()), "live": int(lv.sum()),
                  "live_frac": float(lv.mean()),
                  "reach_p50_m": float(np.median(np.maximum(ell, dz))),
                  "reach_p99_m": float(np.quantile(np.maximum(ell, dz), 0.99))}
        print(f"  {u:>5}: {int(lv.sum()):>6} / {int(m.sum()):>6} live "
              f"({100 * lv.mean():>5.2f}%)   "
              f"reach p50 {per[u]['reach_p50_m']:.2e} m  "
              f"p99 {per[u]['reach_p99_m']:.2e} m")

    out["strata"] = per
    out["live_frac"] = float(live.mean())
    out["inert_frac"] = float(1.0 - live.mean())

    print(f"\n  N = {a.n} at reach >= {a.reach} m: "
          f"{int(live.sum())} live, {int((~live).sum())} inert "
          f"-> {100 * (1 - live.mean()):.2f}% of the PDE budget is spent "
          f"where\n  the Richards residual is C* dpsi*/dt* = 0 and any frozen "
          f"field satisfies it exactly.")

    # ---- 3. where the live points actually are ---------------------------
    if live.any():
        zl = z_phys[live]
        print(f"\n  live points span z = {zl.min():.1f} .. {zl.max():.1f} m "
              f"(domain is {g.Z_BASE:.0f} .. 358 m)")
        out["live_z_min"], out["live_z_max"] = float(zl.min()), float(zl.max())

    # ---- 4. the front-arrival classifier, over (x, z, t) -----------------
    #
    # Section 3 above is a t = 0 picture: it asks which points are live ON THE
    # INITIAL CONDITION. That is not the same question as which points are live
    # during the run, because the rain BC injects a front at the ground surface
    # and the wetted zone behind it is conductive.
    #
    # Kinematic-wave arrival: with the capacity-limited flux q = min(rain.n,
    # K_s) and a sharp front, the wetting front descends at v = q / dtheta.
    # A point at depth d below the ground surface is reached at t = d / v.
    # Points with t < t_arrival are still on the IC and inert; points with
    # t >= t_arrival sit in the wetted zone where K* is orders larger.
    #
    # This is a screening estimate, not a solution -- a real front is diffuse
    # and decelerates. It is deliberately GENEROUS (sharp front, no capillary
    # retardation), so the live fractions it reports are upper bounds.
    print("\n--- front-arrival classifier over (x, z, t) ---")
    from src.step21_geometry import boundaries as bnd
    x_phys = coll.x.detach().reshape(-1).numpy() * s.L_ref
    t_days = coll.t.detach().reshape(-1).numpy()
    depth = g.z_ground(x_phys) - z_phys

    wetted = np.zeros(len(z_phys), dtype=bool)
    arr = {}
    for u in sorted(set(coll.tag.tolist())):
        m = coll.tag == u
        mat = mats[u]
        q = min(bnd.RAIN_FLUX, mat.K_s)              # capacity-limited
        v_m_per_day = q / (mat.theta_s - mat.theta_r) * 86400.0
        t_arr = depth[m] / v_m_per_day
        wetted[m] = t_days[m] >= t_arr
        arr[u] = {"q_applied": q, "v_m_per_day": float(v_m_per_day),
                  "front_30d_m": float(v_m_per_day * a.t_days),
                  "wetted_frac": float(wetted[m].mean())}
        print(f"  {u:>5}: q = {q:.3e} m/s, v = {v_m_per_day:>9.4f} m/d "
              f"-> front reaches {v_m_per_day * a.t_days:>8.2f} m in "
              f"{a.t_days:g} d;  {100 * wetted[m].mean():>5.2f}% of its "
              f"points are behind the front")

    out["front"] = arr
    out["wetted_frac"] = float(wetted.mean())
    either = wetted | live
    out["live_or_wetted_frac"] = float(either.mean())
    print(f"\n  live on the IC OR behind the front: "
          f"{100 * either.mean():.2f}% of the {a.n} points.")
    print("  The two classifiers disagree because they answer different "
          "questions:\n  section 3 is where the physics starts, section 4 is "
          "where it goes.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())