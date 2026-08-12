"""Loss terms for the coupled hydro-mechanical PINN.

PINN.forward(x, z, t) takes three separate (N, 1) leaf tensors and
returns a stacked (N, 3) tensor with columns [psi*, u*, v*]. The slicing
helpers below exist because residuals._psi_of mishandles that shape (it
takes the isinstance(Tensor) branch and returns all three columns).
Slicing lives here rather than in residuals.py to keep the
mutation-tested Richards path untouched.
"""

from __future__ import annotations

import numpy as np
import torch  # type: ignore

from src.nondim import SCALES, Scales
from src.residuals import richards_residual, darcy_flux, interface_flux_jump
from src.sampling import load_initial
from src.step21_geometry import boundaries as bnd


def psi_of(out: torch.Tensor) -> torch.Tensor:
    """Pressure head column, shape (N, 1)."""
    return out[:, 0:1]


def uv_of(out: torch.Tensor) -> torch.Tensor:
    """Displacement columns (u*, v*), shape (N, 2)."""
    return out[:, 1:3]


def ic_targets(path: str | None = None, s: Scales = SCALES):
    """(Collocation at t*=0, psi0_star) with psi0 non-dimensionalised.

    load_initial returns `targets` as a raw passthrough of the cache --
    coordinates are divided by L_ref inside it, but psi0 is NOT. It is
    stored in metres, so it must be divided by H_ref here to be
    comparable with the network's psi* output. Omitting this is a
    factor-30 error.
    """
    coll, targets = load_initial() if path is None else load_initial(path)

    psi0 = torch.as_tensor(
        targets["psi0"], dtype=coll.x.dtype, device=coll.x.device
    ).reshape(-1, 1)

    return coll, (psi0 / s.H_ref).detach()

def L_IC(net, coll, psi0_star):
    """mean_w[(psi* - psi0*)^2] + mean_w[u*^2 + v*^2] at t* = 0.

    Displacements are homogeneous at t = 0, so the mechanical term needs
    no target data. sig_v/sig_h in the cache are the sigma_0 lookup for
    the mechanical residual, not IC targets, so this term does not wait
    on the gravity warm-up.

    `coll.w` is already mean-normalised in load_initial, so dividing by
    its sum gives a true weighted mean comparable to an unweighted MSE.
    """
    out = net(coll.x, coll.z, coll.t)

    w = coll.w.detach()
    wsum = w.sum()

    r_psi = psi_of(out) - psi0_star
    head = (w * r_psi.pow(2)).sum() / wsum

    uv = uv_of(out)
    disp = (w * uv.pow(2).sum(dim=1, keepdim=True)).sum() / wsum

    return head + disp, {"ic_head": head.detach(), "ic_disp": disp.detach()}


# ---------------------------------------------------------------
# 3. PDE loss — hydro half
#    Mechanical half blocked on the gravity warm-up (residuals.py:215)
# ---------------------------------------------------------------
def _psi_field(net):
    """Wrap `net` so it yields psi* only, shape (N, 1).

    residuals._psi_of returns a bare Tensor unchanged, so handing it the
    raw net would make it treat u*, v* as extra psi columns and sum the
    gradients across all three. No exception -- just a wrong number.
    """
    def fields(x, z, t):
        return psi_of(net(x, z, t))
    return fields


def L_PDE(net, coll, mats, s: Scales = SCALES, normalise: str = "none",
          per_tag: bool = False):
    """mean_w[R_Richards^2] over the interior collocation set.

    Split by coll.tag: richards_residual takes ONE material, and Mk /
    Mk_d / Tm differ in K_s by ~3 orders. Masked blending is not an
    option -- it would evaluate every material's van Genuchten at every
    point and 0 * NaN = NaN would take the whole loss down silently.

    Squared residuals accumulate across tags and divide by the GLOBAL
    weight sum. Averaging the three per-tag means instead would give the
    0.25-share material equal say with the 0.40-share one.

    mats: {tag -> material}, from materials.py. Passed in, not inlined,
    so the tag -> material mapping has one source of truth.
    """
    fields = _psi_field(net)
    w_all = coll.w.detach()
    wsum = w_all.sum()

    total = torch.zeros((), dtype=coll.x.dtype, device=coll.x.device)
    parts = {}

    for tag in sorted(set(coll.tag.tolist())):
        if tag not in mats:
            raise KeyError(
                f"L_PDE: collocation tag {tag!r} has no entry in mats. "
                f"Tags present: {sorted(set(coll.tag.tolist()))}"
            )
        m = torch.as_tensor(coll.tag == tag, device=coll.x.device)
        xs, zs, ts = coll.x[m], coll.z[m], coll.t[m]

        R = richards_residual(fields, xs, zs, ts, mats[tag],
                              s=s, normalise=normalise)

        contrib = (w_all[m] * R.pow(2)).sum()
        total = total + contrib
        if per_tag:
            parts[f"pde_richards_{tag}"] = (contrib / w_all[m].sum()).detach()

    pde = total / wsum
    return pde, {"pde_richards": pde.detach(), **parts}

# Segment -> condition type. Explicit, not inferred: every segment is named
# exactly once, so adding a seventh is a KeyError rather than a silent skip.
_BC_KIND = {
    "natural_ground": "flux",
    "bench":          "flux",
    "cut_face":       "flux",      # zero while SEEPAGE_FACE_MODE == "noflow"
    "pit_floor":      "flux",      # zero: water table is below the domain
    "base":           "flux",      # zero
    "far_field_f1":   "dirichlet",
}


def L_BC(net, bcs, mats, s: Scales = SCALES, per_segment: bool = False):
    """Boundary loss over the six segments. Hydraulic only.

    bcs : {segment -> Collocation}, from sample_boundary. Requires nx/nz.
    mats: {tag -> Material}, as L_PDE.

    SIGN CONVENTION -- the one thing to get right here.
        boundaries.flux_bc returns POSITIVE = INTO the domain.
        darcy_flux returns q*, and n is the OUTWARD normal, so q*.n is
        positive OUT of the domain.
        The target is therefore  q*.n = -q_prescribed*.
        Dropping that minus sign makes rainfall drain the slope instead of
        wetting it. The loss still converges; the physics is inverted.

    Accumulates squared residuals and divides by the GLOBAL weight sum, the
    same convention as L_PDE. Segments therefore contribute in proportion to
    their point count. Since sample_boundaries allocates n, n//4, n//3, n//2
    by segment, that weighting is a CHOICE inherited from the sampler, not a
    physical statement -- record it in the decision log and revisit if the
    far-field condition turns out to dominate.
    """
    fields = _psi_field(net)
    total = None
    wsum = 0.0
    parts = {}

    for seg, coll in bcs.items():
        if seg not in _BC_KIND:
            raise KeyError(
                f"L_BC: segment {seg!r} has no condition type. "
                f"Known: {sorted(_BC_KIND)}")
        kind = _BC_KIND[seg]
        w = coll.w.detach()

        if kind == "dirichlet":
            # psi prescribed on F1: hydrostatic w.r.t. a water table at Z_WT,
            # which lies BELOW the domain, so this is negative everywhere.
            # fields() already yields (N,1) psi* -- see _psi_field.
            z_phys = (coll.z * s.L_ref).detach().cpu().numpy().ravel()
            psi_target = torch.as_tensor(
                bnd.psi_far_field(z_phys) / s.H_ref,
                dtype=coll.x.dtype, device=coll.x.device).reshape(-1, 1)
            R = fields(coll.x, coll.z, coll.t) - psi_target

        else:
            if coll.nx is None or coll.nz is None:
                raise ValueError(
                    f"L_BC: segment {seg!r} carries no normals. Rebuild the "
                    f"boundary set with the current sample_boundary.")

            # Prescribed flux, per point, in m/s. Positive = into the domain.
            # Goes through boundaries.flux_bc so the K_s capacity cap applies
            # here exactly as it does in the geometry layer -- one source of
            # truth for what the ground can actually accept.
            pts = np.column_stack([
                (coll.x * s.L_ref).detach().cpu().numpy().ravel(),
                (coll.z * s.L_ref).detach().cpu().numpy().ravel()])
            q_in = bnd.flux_bc(seg, pts, tag=coll.tag) / s.Q_ref
            q_target = torch.as_tensor(
                -q_in, dtype=coll.x.dtype,
                device=coll.x.device).reshape(-1, 1)        # outward-positive

            # Split by material tag: darcy_flux takes ONE material, and K_s
            # spans three orders across the strata. Same reasoning as L_PDE.
            R = torch.zeros_like(coll.x)
            for tag in sorted(set(coll.tag.tolist())):
                if tag not in mats:
                    raise KeyError(
                        f"L_BC: boundary tag {tag!r} on segment {seg!r} has "
                        f"no entry in mats. Present: {sorted(mats)}")
                m = torch.as_tensor(coll.tag == tag, device=coll.x.device)
                idx = torch.as_tensor(
                    np.flatnonzero(coll.tag == tag),
                    dtype=torch.long, device=coll.x.device)
                qx, qz = darcy_flux(fields, coll.x[idx], coll.z[idx],
                                    coll.t[idx], mats[tag], s)
                qn = qx * coll.nx[idx] + qz * coll.nz[idx]
                R = R.index_copy(0, idx, qn - q_target[idx])

        contrib = (w * R.pow(2)).sum()
        total = contrib if total is None else total + contrib
        wsum += w.sum().item()
        if per_segment:
            parts[f"bc_{seg}"] = (contrib / w.sum()).detach()

    bc = total / wsum
    return bc, {"bc": bc.detach(), **parts}

# ===========================================================================
# Interface flux-continuity loss.
#
# A single network gives psi continuity across a contact for free; what it does
# not give is mass conservation ACROSS it. See `interface_flux_jump` for why
# nothing else in the loss can see the leak.
#
# Contact sets come from `sample_interfaces`, keyed "mk_tm" / "mkd_tm". Tm is
# the lower material at both by construction of that sampler, so mat_b is
# always Tm and mat_a comes from the tag that travelled with the Collocation.
# ===========================================================================

_INTERFACE_LOWER = "Tm"


def L_interface(net, ifaces, mats, s: Scales = SCALES,
                per_contact: bool = False):
    """Squared normal-flux jump across the Mk|Tm and Mk_d|Tm contacts.

    Returns `(scalar, {"interface": ..., **per_contact_parts})`, matching
    `L_BC`: the TOTAL divides by the global weight sum, while each per-contact
    part is a per-contact mean. The two are NOT additive -- see DECISIONS.md.

    w_interface = 1.0 needs no normalisation factor: the measured jump/flux
    ratio is 1.119 and |q*| ~ 2.8e-3, the same order as L_PDE at ~3e-3.
    """
    fields = _psi_field(net)
    W = sum(float(c.w.sum()) for c in ifaces.values())

    total = None
    parts = {}

    for key, coll in ifaces.items():
        upper = np.unique(coll.tag)
        assert upper.size == 1, f"{key} mixes materials above the contact: {upper}"
        mat_a = mats[str(upper[0])]
        mat_b = mats[_INTERFACE_LOWER]

        jump = interface_flux_jump(fields, coll.x, coll.z, coll.t,
                                   mat_a, mat_b, coll.nx, coll.nz, s)

        contrib = (coll.w * jump.pow(2)).sum()
        total = contrib if total is None else total + contrib
        if per_contact:
            parts[f"interface_{key}"] = (contrib / coll.w.sum()).detach()

    interface = total / W
    return interface, {"interface": interface.detach(), **parts}
