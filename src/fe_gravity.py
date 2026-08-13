"""
src/fe_gravity.py — offline linear-elastic gravity solve for sigma_0.
=====================================================================

Step 3.3a. Produces an EQUILIBRATED initial stress field by actually solving

    div sigma + rho_b g = 0,   sigma = D(E,nu):eps,   eps = sym grad u

on the Section 5 domain, rather than assuming sigma_h = K0 sigma_v.

WHY THE K0 COLUMN CANNOT BE FIXED BY TUNING K0
----------------------------------------------
The K0 column sets sigma_xz = 0 everywhere. Horizontal equilibrium is then
d(sigma_xx)/dx = 0, which under a sloping ground surface is false: sigma_v
varies with x, so sigma_h = K0*sigma_v does too. Measured on the Step 2.3 IC
cache the imbalance is 0.128 rho*g mean, 0.254 in the Mk wedge. No choice of
K0 removes it — a geostatic state under a slope MUST carry shear. That is
what this module computes.

METHOD
------
Constant-strain triangles (CST), plane strain, direct sparse solve. CST is
deliberate: stress is element-wise constant, which is exactly the quantity
wanted, and there is no superconvergence story to get wrong. The mesh is
structured and interface-conforming, so the usual CST complaint (locking on
distorted meshes) does not bite.

Mesh: two transfinite blocks.
  A  x in [X_MIN, X_MK_DIVIDE], vertical columns.
  B  x from X_MK_DIVIDE to the F1 trace, columns slanting with F1.
The block seam is the vertical Mk|Mk_d contact, so no element straddles the
5.5x modulus jump between Mk (2.09e8) and Mk_d (3.806e7). Within each column
a node is placed exactly on the Tm contact, so no element straddles the 20x
jump to Tm (4.264e9) either. Elements are assigned a material by centroid.

BOUNDARY CONDITIONS (from boundaries.mechanical_bc)
  base          z = Z_BASE          u = v = 0
  pit_floor     x = X_MIN           u = 0     (roller)
  far_field_f1  x = x_f1(z)         u = 0     (roller)
  natural_ground / bench / cut_face traction-free (natural in FE; no term)

SIGN CONVENTION: TENSION POSITIVE, matching src/mechanics.py and D-3.3.1.
`boundaries.sigma_v_geostatic` is compression-positive; do not mix them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

__all__ = ["Mesh", "build_mesh", "plane_strain_D", "assemble", "solve_gravity",
           "element_stress", "nodal_stress"]


# ---------------------------------------------------------------------------
# 1. Mesh
# ---------------------------------------------------------------------------
@dataclass
class Mesh:
    nodes: np.ndarray        # (N,2) x,z
    tris: np.ndarray         # (M,3) node indices, CCW
    tag: np.ndarray          # (M,) material tag per element
    area: np.ndarray         # (M,)

    @property
    def centroids(self) -> np.ndarray:
        return self.nodes[self.tris].mean(axis=1)


def _column(x_bot, x_top, z_bot, z_top_fn, n_below, n_above, contact_fn):
    """Node elevations along one column, with a node exactly on the contact.

    The column is the straight segment (x_bot, z_bot) -> (x_top, z_top).
    Returns (xs, zs) of length n_below + n_above + 1.
    """
    z_top = z_top_fn(x_top)

    def pt(s):
        return x_bot + s * (x_top - x_bot), z_bot + s * (z_top - z_bot)

    # s where the column crosses the Tm contact
    def f(s):
        x, z = pt(s)
        return z - contact_fn(x)

    n_tot = n_below + n_above
    if f(0.0) >= 0.0 or f(1.0) <= 0.0:
        # Contact lies outside this column: it is entirely one material. Do
        # NOT keep a split, or the n_below+1 lower nodes collapse onto s = 0
        # and every element in that strip has zero area. That produced a
        # silent NaN in the first patch test: K0 still came out right because
        # the degenerate rows contributed nothing, so the answer looked fine
        # while a third of the mesh did not exist.
        s = np.linspace(0.0, 1.0, n_tot + 1)
    else:
        s_c = brentq(f, 0.0, 1.0, xtol=1e-12)
        s = np.concatenate([np.linspace(0.0, s_c, n_below + 1),
                            np.linspace(s_c, 1.0, n_above + 1)[1:]])
    xs, zs = pt(s)

    # Snap the endpoints. pt(1.0) computes z_bot + 1.0*(z_top - z_bot), which
    # in IEEE is not guaranteed to return exactly z_top -- it can land 1 ulp
    # above the ground surface, and `inside_domain` then says "outside".
    # Whether it does depends on the numpy build, so without this the mesh is
    # not reproducible across machines. Same failure mode as the
    # x -> x/L_ref -> x round trip that forced geometry queries to happen once,
    # at sample time.
    xs[0], zs[0] = x_bot, z_bot
    xs[-1], zs[-1] = x_top, z_top
    return xs, zs


def build_mesh(g, n_col_a=24, n_col_b=40, n_below=26, n_above=14):
    """Two-block structured triangulation of the Section 5 domain."""
    # top of the F1 edge: z where z == z_ground(x_f1(z))
    z_top_f1 = brentq(lambda z: z - g.z_ground(g.x_f1(z)), g.Z_BASE + 1.0, 400.0)
    x_top_f1 = float(g.x_f1(z_top_f1))
    x_bot_f1 = float(g.x_f1(g.Z_BASE))

    def contact(x):
        return min(float(g.z_tm_top(x)), float(g.z_ground(x)))

    def ground(x):
        return float(g.z_ground(x))

    cols = []
    for i in range(n_col_a + 1):                       # block A, vertical
        xi = i / n_col_a
        x = g.X_MIN + xi * (g.X_MK_DIVIDE - g.X_MIN)
        cols.append(_column(x, x, g.Z_BASE, ground, n_below, n_above, contact))
    for i in range(1, n_col_b + 1):                    # block B, slanted
        xi = i / n_col_b
        xb = g.X_MK_DIVIDE + xi * (x_bot_f1 - g.X_MK_DIVIDE)
        xt = g.X_MK_DIVIDE + xi * (x_top_f1 - g.X_MK_DIVIDE)
        cols.append(_column(xb, xt, g.Z_BASE, ground, n_below, n_above, contact))

    nz = n_below + n_above + 1
    nodes = np.empty((len(cols) * nz, 2))
    for j, (xs, zs) in enumerate(cols):
        nodes[j * nz:(j + 1) * nz, 0] = xs
        nodes[j * nz:(j + 1) * nz, 1] = zs

    tris = []
    for j in range(len(cols) - 1):
        for k in range(nz - 1):
            a = j * nz + k
            b = (j + 1) * nz + k
            tris.append((a, b, b + 1))
            tris.append((a, b + 1, a + 1))
    tris = np.asarray(tris, dtype=np.int64)

    p = nodes[tris]
    area = 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                  - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    flip = area < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    area = np.abs(area)

    c = nodes[tris].mean(axis=1)
    tag = g.material_tag(c[:, 0], c[:, 1])
    return Mesh(nodes=nodes, tris=tris, tag=np.asarray(tag), area=area)


# ---------------------------------------------------------------------------
# 2. Elasticity
# ---------------------------------------------------------------------------
def plane_strain_D(E: float, nu: float) -> np.ndarray:
    f = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return f * np.array([[1.0 - nu, nu, 0.0],
                         [nu, 1.0 - nu, 0.0],
                         [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]])


def _B(p):
    """(3,6) strain-displacement for one CST. p is (3,2)."""
    (x1, z1), (x2, z2), (x3, z3) = p
    A2 = (x2 - x1) * (z3 - z1) - (x3 - x1) * (z2 - z1)
    b = np.array([z2 - z3, z3 - z1, z1 - z2]) / A2
    c = np.array([x3 - x2, x1 - x3, x2 - x1]) / A2
    B = np.zeros((3, 6))
    B[0, 0::2] = b
    B[1, 1::2] = c
    B[2, 0::2] = c
    B[2, 1::2] = b
    return B


def _per_element(spec, mesh: Mesh):
    """Resolve a property given as a callable-of-tag OR an (M,) array.

    rho_b is NOT a function of material alone: it is
    rho_dry + theta(psi_0)*rho_w, so it varies within a stratum with the
    initial suction. Passing it as an array is the normal case, not the
    exception. E and nu are tag-uniform and normally arrive as callables.
    """
    a = np.asarray(spec)
    if a.dtype != object and a.ndim == 1:
        if len(a) != len(mesh.tris):
            raise ValueError(f"per-element array has {len(a)} entries for "
                             f"{len(mesh.tris)} elements")
        return a
    return np.array([spec(t) for t in mesh.tag], dtype=float)


def assemble(mesh: Mesh, E_of, nu_of, rho_of, g_acc=9.81):
    """Global stiffness (sparse) and gravity load vector. Gravity acts in -z."""
    E_e = _per_element(E_of, mesh)
    nu_e = _per_element(nu_of, mesh)
    rho_e = _per_element(rho_of, mesh)
    n = len(mesh.nodes)
    rows, cols, vals = [], [], []
    F = np.zeros(2 * n)

    for e in range(len(mesh.tris)):
        idx = mesh.tris[e]
        p = mesh.nodes[idx]
        D = plane_strain_D(E_e[e], nu_e[e])
        B = _B(p)
        A = mesh.area[e]
        Ke = A * (B.T @ D @ B)

        dofs = np.empty(6, dtype=np.int64)
        dofs[0::2] = 2 * idx
        dofs[1::2] = 2 * idx + 1
        rows.append(np.repeat(dofs, 6))
        cols.append(np.tile(dofs, 6))
        vals.append(Ke.ravel())

        fz = -rho_e[e] * g_acc * A / 3.0
        np.add.at(F, 2 * idx + 1, fz)

    K = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * n, 2 * n)).tocsr()
    return K, F


def solve_gravity(mesh: Mesh, E_of, nu_of, rho_of, fixed_dofs, g_acc=9.81):
    """Solve K u = F with the given Dirichlet dofs held at zero."""
    K, F = assemble(mesh, E_of, nu_of, rho_of, g_acc)
    n2 = K.shape[0]
    free = np.setdiff1d(np.arange(n2), np.asarray(fixed_dofs, dtype=np.int64))
    u = np.zeros(n2)
    u[free] = spsolve(K[free][:, free].tocsc(), F[free])
    return u


# ---------------------------------------------------------------------------
# 3. Recovery
# ---------------------------------------------------------------------------
def element_stress(mesh: Mesh, u, E_of, nu_of):
    """(M,3) [sigma_xx, sigma_zz, sigma_xz], TENSION POSITIVE."""
    E_e = _per_element(E_of, mesh)
    nu_e = _per_element(nu_of, mesh)
    out = np.empty((len(mesh.tris), 3))
    for e in range(len(mesh.tris)):
        idx = mesh.tris[e]
        dofs = np.empty(6, dtype=np.int64)
        dofs[0::2] = 2 * idx
        dofs[1::2] = 2 * idx + 1
        D = plane_strain_D(E_e[e], nu_e[e])
        out[e] = D @ (_B(mesh.nodes[idx]) @ u[dofs])
    return out


def nodal_stress(mesh: Mesh, sig_e):
    """Area-weighted nodal average of the element stresses, for interpolation.

    Averaging ACROSS a material contact is wrong — stress is genuinely
    discontinuous there — so nodes on a contact get the average of the
    elements sharing their own material only where that is unambiguous. Here
    we average everything and accept the smearing, because sigma_0 is used as
    a smooth background field, not as a stress-jump diagnostic. Do not use
    these values to check interface conditions.
    """
    n = len(mesh.nodes)
    acc = np.zeros((n, 3))
    wsum = np.zeros(n)
    for k in range(3):
        np.add.at(acc[:, k], mesh.tris.ravel(),
                  np.repeat(sig_e[:, k] * mesh.area, 3))
    np.add.at(wsum, mesh.tris.ravel(), np.repeat(mesh.area, 3))
    return acc / wsum[:, None]


# ---------------------------------------------------------------------------
# 4. Evaluation at arbitrary points
# ---------------------------------------------------------------------------
def sigma0_interpolator(mesh: Mesh, sig_e):
    """Callable (x, z) -> (M,3) sigma_0 in Pa, TENSION POSITIVE.

    Uses matplotlib's LinearTriInterpolator over THIS triangulation rather
    than scipy's LinearNDInterpolator, which would re-triangulate and bridge
    the concave notch at the cut face — producing stress inside the pit, where
    there is no rock.

    Points outside the mesh come back as NaN. That is deliberate: a silent
    zero would look like a stress-free region rather than a sampling bug.
    """
    from matplotlib.tri import LinearTriInterpolator, Triangulation

    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    sig_n = nodal_stress(mesh, sig_e)
    interps = [LinearTriInterpolator(tri, sig_n[:, k]) for k in range(3)]

    def sigma0(x, z):
        x = np.asarray(x, float)
        z = np.asarray(z, float)
        out = np.empty(x.shape + (3,))
        for k, f in enumerate(interps):
            v = f(x, z)
            out[..., k] = np.where(np.ma.getmaskarray(v), np.nan, np.ma.filled(v, np.nan))
        return out

    return sigma0
