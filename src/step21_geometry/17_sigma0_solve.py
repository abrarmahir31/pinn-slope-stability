"""Step 3.3a — solve the elastic gravity turn-on and cache sigma_0.

Acceptance checks in the same style as 14_ic_checks.py: every check guards
against passing on an empty array, and the script writes nothing if a check
fails.

Output: sigma0_cache.npz next to ic_cache.npz. Gitignored and deterministic,
so regenerating is the intended path rather than shipping the binary.
"""
# Windows: torch must load before numpy/MKL or shm.dll fails to resolve.
import torch as _torch  # noqa: F401
import hashlib
import pathlib
import sys

import numpy as np

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[2])]

import geometry as g
import boundaries as b
from initial import psi_initial
from src import materials as mt
from src.fe_gravity import (build_mesh, element_stress, nodal_stress,
                            sigma0_interpolator, solve_gravity, assemble)

N_FAIL = 0
G_ACC = 9.81

# Mesh resolution. mean|sigma_xz| changes by 0.2% between this and a mesh
# 2.25x finer; max|sigma_xz| does NOT converge, because the re-entrant corner
# at the toe is a genuine elastic singularity. Report medians, not maxima.
NA, NB, NBELOW, NABOVE = 30, 50, 26, 16


def check(label, cond, detail=""):
    global N_FAIL
    ok = bool(cond)
    if not ok:
        N_FAIL += 1
    print(f"[{'PASS' if ok else 'FAIL'}] {label:46s} {detail}")


def geometry_hash():
    p = pathlib.Path(__file__).with_name("geometry.py")
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def main():
    M = mt.load_materials()
    mesh = build_mesh(g, NA, NB, NBELOW, NABOVE)
    c = mesh.centroids

    check("mesh has no degenerate elements", mesh.area.min() > 1e-6,
          f"min area {mesh.area.min():.4g} m2")
    check("no element centroid outside the domain",
          (mesh.tag == "outside").sum() == 0,
          f"{len(mesh.tris)} elements")
    # Boundary nodes sit ON the boundary by construction, so an exact
    # inside_domain test is a floating-point coin toss: it passed on one numpy
    # build and failed on 3 of 3483 nodes on another. What matters is the
    # MAGNITUDE of any excursion, not its sign -- 1 ulp outside a surface
    # digitised to 0.149 m RMS is meaningless, 1 mm is not.
    nx, nz_ = mesh.nodes[:, 0], mesh.nodes[:, 1]
    excursion = np.maximum.reduce([nz_ - g.z_ground(nx), g.Z_BASE - nz_,
                                   nx - g.x_f1(nz_), g.X_MIN - nx])
    check("no node outside the domain beyond tolerance",
          excursion.max() < 1e-6,
          f"max excursion {excursion.max():.3e} m, "
          f"{(excursion > 0).sum()}/{len(mesh.nodes)} nodes on the far side")
    check("all three units meshed",
          all((mesh.tag == k).any() for k in ("Mk", "Mk_d", "Tm")),
          str({k: int((mesh.tag == k).sum()) for k in ("Mk", "Mk_d", "Tm")}))

    psi_c = psi_initial(c[:, 0], c[:, 1])
    rho_e = np.array([M[t].rho_dry + mt.theta(psi_c[i], M[t]) * 1000.0
                      for i, t in enumerate(mesh.tag)])
    E_e = np.array([M[t].E for t in mesh.tag])
    nu_e = np.array([M[t].nu for t in mesh.tag])
    check("bulk density is wet and in range",
          (rho_e.min() > 1500.0) and (rho_e.max() < 2800.0),
          f"{rho_e.min():.1f} .. {rho_e.max():.1f} kg/m3")

    # --- boundary conditions, from boundaries.mechanical_bc ---
    nd = mesh.nodes
    nz = NBELOW + NABOVE + 1
    ncols = len(nd) // nz
    fixed = []
    base = np.flatnonzero(np.abs(nd[:, 1] - g.Z_BASE) < 1e-7)
    fixed += list(2 * base) + list(2 * base + 1)          # base: u = v = 0
    pit = np.flatnonzero(np.abs(nd[:, 0] - g.X_MIN) < 1e-7)
    fixed += list(2 * pit)                                # pit_floor: roller
    f1 = np.arange((ncols - 1) * nz, ncols * nz)
    fixed += list(2 * f1)                                 # far_field_f1: roller
    fixed = sorted(set(fixed))
    check("all three constrained segments found",
          len(base) > 10 and len(pit) > 10 and len(f1) > 10,
          f"base {len(base)}, pit_floor {len(pit)}, F1 {len(f1)}")

    u = solve_gravity(mesh, E_e, nu_e, rho_e, fixed, G_ACC)
    sig = element_stress(mesh, u, E_e, nu_e)

    # --- the acceptance check that matters: global equilibrium ---
    K, F = assemble(mesh, E_e, nu_e, rho_e, G_ACC)
    R = K @ u - F
    W = float((rho_e * mesh.area * G_ACC).sum())
    check("vertical reactions equal the total weight",
          abs(R[1::2].sum() / W - 1.0) < 1e-9,
          f"R_z/W = {R[1::2].sum()/W:.12f}")
    check("horizontal reactions sum to zero",
          abs(R[0::2].sum()) / W < 1e-12,
          f"R_x = {R[0::2].sum():+.3e} N/m")

    sv, sxz = -sig[:, 1], sig[:, 2]      # compression-positive sigma_v
    depth = g.z_ground(c[:, 0]) - c[:, 1]
    deep = depth > 20.0
    sv_col = b.sigma_v_geostatic(c[deep, 0], c[deep, 1], wet=True)
    check("sigma_v tracks the column integral at depth",
          abs(np.median(sv[deep] / sv_col) - 1.0) < 0.10,
          f"median FE/column = {np.median(sv[deep]/sv_col):.4f}")
    check("compressive except at the free surface",
          (sv < 0).sum() < 0.01 * len(sv),
          f"{(sv<0).sum()} tensile of {len(sv)}, max depth "
          f"{depth[sv<0].max() if (sv<0).any() else 0.0:.2f} m")
    check("the equilibrated state carries real shear",
          np.median(np.abs(sxz[deep]) / sv_col) > 0.02,
          f"median |sxz|/sigma_v = {np.median(np.abs(sxz[deep])/sv_col):.4f}")

    if N_FAIL:
        print(f"\n{N_FAIL} CHECK(S) FAILED -- nothing written")
        return 1

    out = pathlib.Path(__file__).with_name("sigma0_cache.npz")
    np.savez(out,
             nodes=mesh.nodes, tris=mesh.tris, tag=mesh.tag, area=mesh.area,
             sigma_e=sig, sigma_n=nodal_stress(mesh, sig), u=u,
             rho_e=rho_e, E_e=E_e, nu_e=nu_e,
             meta=np.array([geometry_hash(), f"{NA},{NB},{NBELOW},{NABOVE}",
                            "tension_positive", f"W={W:.6e}"]))
    print(f"\nALL CHECKS PASSED -- wrote {out.name} "
          f"({len(mesh.tris)} elements, geometry hash {geometry_hash()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
