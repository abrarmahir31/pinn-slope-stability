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
from src.coupling import KC_DEFAULT, KC_OFF, KCConfig, kozeny_carman_factor
from src.mechanics import mechanical_residual, strain_star, total_stress_star
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

def L_IC(net, coll, psi0_star, keep_graph: bool = False):
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

    d = (lambda v: v) if keep_graph else (lambda v: v.detach())
    return head + disp, {"ic_head": d(head), "ic_disp": d(disp)}


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


def _kc_factor(net, x, z, t, mat, kc: KCConfig, tag: str, s: Scales):
    """Kozeny-Carman conductivity multiplier at these points, or None.

    None -- not a tensor of ones -- when the feedback is off for this stratum.
    `richards_residual` then takes the identical code path it took before the
    argument existed, which is what makes the one-way arm of the Step 6.2
    ablation a genuine reproduction rather than a numerically-close rerun.

    eps_v is requested in PHYSICAL units. Kozeny-Carman consumes true strain;
    eps_v* is larger by L_ref/U_ref = 100, and passing the wrong one inflates
    the feedback hundredfold while changing nothing else visible.

    The returned tensor carries its graph. That graph IS the coupling: it is
    the path by which the Richards residual comes to depend on u*, v*.
    """
    if not kc.is_on(tag):
        return None
    u, v = uv_of(net(x, z, t)).split(1, dim=1)
    eps_v = strain_star(u, v, x, z, physical=True, s=s)[3]
    return kozeny_carman_factor(mat.n0, eps_v, kc, tag=tag)


def L_PDE(net, coll, mats, s: Scales = SCALES, normalise: str = "none",
          per_tag: bool = False, kc: KCConfig = KC_OFF):
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
                              s=s, normalise=normalise,
                              k_factor=_kc_factor(net, xs, zs, ts, mats[tag],
                                                  kc, tag, s))

        contrib = (w_all[m] * R.pow(2)).sum()
        total = total + contrib
        if per_tag:
            parts[f"pde_richards_{tag}"] = (contrib / w_all[m].sum()).detach()

    pde = total / wsum
    return pde, {"pde_richards": pde.detach(), **parts}

def L_PDE_mech(net, coll, mats, s: Scales = SCALES, per_tag: bool = False,
               bishop: bool = True):
    """mean_w[R_mech_x^2 + R_mech_z^2] over the interior collocation set.

    Requires `coll.sigma0` and `coll.rho0`, attached by
    `sigma0.attach_sigma0`. Both are checked here rather than defaulted,
    because the defaults that would otherwise apply are silently wrong:
    sigma0 = 0 is a stress-free domain, and rho0 = 1 is rho_b_ref rather than
    the wet profile the FE solve was equilibrated against.

    Tag-partitioned for the same reason as L_PDE: E spans 3.81e7 to 4.264e9
    across the three strata, and `mechanical_residual` takes ONE material.

    The x and z components are summed, not averaged. They are the two
    components of one vector equation, so weighting them separately would be
    a modelling choice with nothing behind it.
    """
    if coll.sigma0 is None or coll.rho0 is None:
        raise ValueError(
            "L_PDE_mech: this Collocation has no sigma_0 attached. Call "
            "sigma0.attach_sigma0(coll) at sample time. Running without it "
            "would silently solve a stress-free, wrong-density problem."
        )

    w_all = coll.w.detach()
    wsum = w_all.sum()
    total = torch.zeros((), dtype=coll.x.dtype, device=coll.x.device)
    parts = {}

    for tag in sorted(set(coll.tag.tolist())):
        if tag not in mats:
            raise KeyError(
                f"L_PDE_mech: collocation tag {tag!r} has no entry in mats. "
                f"Tags present: {sorted(set(coll.tag.tolist()))}"
            )
        m = torch.as_tensor(coll.tag == tag, device=coll.x.device)
        xs, zs, ts = coll.x[m], coll.z[m], coll.t[m]
        sig0 = coll.sigma0[m]
        sigma0_star = (sig0[:, 0:1], sig0[:, 1:2], sig0[:, 2:3])

        res_x, res_z = mechanical_residual(
            net, xs, zs, ts, mats[tag],
            sigma0_star=sigma0_star, rho0_ratio=coll.rho0[m],
            bishop=bishop, s=s)

        contrib = (w_all[m] * (res_x.pow(2) + res_z.pow(2))).sum()
        total = total + contrib
        if per_tag:
            parts[f"pde_mech_{tag}"] = (contrib / w_all[m].sum()).detach()

    mech = total / wsum
    return mech, {"pde_mech": mech.detach(), **parts}


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


def L_BC_mech(net, bcs, mats, s: Scales = SCALES, per_segment: bool = False):
    """Mechanical boundary loss over the six segments.

    Three kinds, from `boundaries.mechanical_bc`:

      base          u* = v* = 0            Dirichlet
      pit_floor     u* = 0                 roller (normal is -x)
      far_field_f1  u* = 0                 roller (normal is +x)
      natural_ground / bench / cut_face    sigma_total . n = 0

    THE TRACTION-FREE CONDITION IS ON TOTAL STRESS, and after D-3.3.3 that is
    a quantity this project can actually assemble. It is the free-surface
    condition: nothing outside the slope pushes on it, and the pore fluid is
    at atmospheric. Both components vanish, so it is two scalar conditions per
    point, not one.

    Without it the network can put any traction it likes on the cut face --
    the boundary the failure mechanism runs through -- while every interior
    residual stays satisfied. That is why D-3.2.5 refused to ship half the
    mechanics: `total_loss` would have looked complete with the one condition
    that constrains the failure surface missing.

    Requires sigma_0 on every boundary Collocation. Boundary sets are nudged
    0.25 m inward for that lookup (see sigma0.attach_sigma0); the roller and
    fixed segments do not use sigma_0 at all, so the approximation touches only
    the three exposed segments.

    Same normalisation as L_BC: squared residuals accumulate and divide by the
    GLOBAL weight sum, so segments contribute in proportion to their point
    count, which is a choice inherited from the sampler.
    """
    total = None
    wsum = 0.0
    parts = {}

    for seg, coll in bcs.items():
        kind = bnd.mechanical_bc(seg)
        w = coll.w.detach()
        wsum = wsum + float(w.sum())

        if kind is None:                       # traction-free
            if coll.sigma0 is None:
                raise ValueError(
                    f"L_BC_mech: segment {seg!r} has no sigma_0 attached. "
                    f"Call sigma0.attach_sigma0 on every boundary set at "
                    f"sample time; a zero sigma_0 here would read as a "
                    f"stress-free slope and be satisfied trivially."
                )
            if coll.nx is None or coll.nz is None:
                raise ValueError(
                    f"L_BC_mech: segment {seg!r} carries no normals.")

            R2 = torch.zeros_like(coll.x)
            for tag in sorted(set(coll.tag.tolist())):
                if tag not in mats:
                    raise KeyError(
                        f"L_BC_mech: boundary tag {tag!r} on segment {seg!r} "
                        f"has no entry in mats. Present: {sorted(mats)}")
                idx = torch.as_tensor(
                    np.flatnonzero(coll.tag == tag),
                    dtype=torch.long, device=coll.x.device)
                # Fresh leaves: the strain here is differentiated w.r.t. these
                # coordinates, and indexing a view silently detaches nothing
                # but makes the graph depend on the parent's layout.
                xs = coll.x[idx].detach().clone().requires_grad_(True)
                zs = coll.z[idx].detach().clone().requires_grad_(True)
                ts = coll.t[idx].detach().clone().requires_grad_(True)
                sg = coll.sigma0[idx]
                (sxx, szz, sxz), _, _ = total_stress_star(
                    net, xs, zs, ts, mats[tag],
                    sigma0_star=(sg[:, 0:1], sg[:, 1:2], sg[:, 2:3]), s=s)

                nx, nz = coll.nx[idx], coll.nz[idx]
                tx = sxx * nx + sxz * nz
                tz = sxz * nx + szz * nz
                R2 = R2.index_copy(0, idx, tx.pow(2) + tz.pow(2))

        else:
            _, cons = kind
            uv = uv_of(net(coll.x, coll.z, coll.t))
            R2 = uv[:, 0:1].pow(2) if "v" not in cons else \
                uv[:, 0:1].pow(2) + uv[:, 1:2].pow(2)

        contrib = (w * R2).sum()
        total = contrib if total is None else total + contrib
        if per_segment:
            parts[f"bcmech_{seg}"] = (contrib / w.sum()).detach()

    bc = total / wsum
    return bc, {"bc_mech": bc.detach(), **parts}


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

# ===========================================================================
# total_loss -- the Step 3.2 deliverable.
#
# On the parts dict. Each term reports its own diagnostics in its own units:
# per-tag / per-segment / per-contact MEANS, so a 100-point segment stays
# comparable with a 400-point one. Those do NOT sum to anything meaningful.
#
# What DOES sum is `contrib_*`: the weighted contribution each term makes to
# the total, w_i * L_i. `total == sum(contrib_*)` exactly, and that identity is
# tested. Read `contrib_*` to see what is driving training; read the per-term
# means to see which part of a term is unhappy.
#
# Mechanics is absent, not disabled-by-default-and-half-implemented. See
# `include_mechanics` below.
# ===========================================================================

DEFAULT_WEIGHTS = {
    "pde": 1.0,
    "pde_mech": 1.0,
    "bc_mech": 1.0,
    "ic": 1.0,
    "bc": 1.0,
    "interface": 1.0,
}


def total_loss(net, coll, bcs, ic, ifaces, mats, s: Scales = SCALES,
               weights: dict | None = None,
               include_mechanics: bool = False,
               include_feedback: bool = False,
               per_term: bool = False):
    """Weighted sum of the hydraulic loss terms.

    Returns `(scalar, parts)`. `parts` always carries the four term totals
    (`pde`, `ic_head`, `ic_disp`, `bc`, `interface`) and the four `contrib_*`
    entries that sum to the scalar. With `per_term=True` it also carries each
    term's own breakdown (per-tag, per-segment, per-contact).

    include_feedback
        Adds the Kozeny-Carman porosity-strain feedback into K*, i.e. the
        hydraulic half of two-way coupling. False gives the one-way arm of the
        Step 6.2 ablation and is bit-for-bit the pre-Step-3.3 hydraulic loss.
        True uses `coupling.KC_DEFAULT` -- feedback on the marls, off for Tm
        (D-3.3.2).

        Independent of `include_mechanics`: feedback needs displacements to be
        meaningful, but it does not need the equilibrium residual to be in the
        loss, and separating them is what makes the ablation a 2x2 rather than
        a single switch.
    include_mechanics
        Adds the equilibrium residual (Step 3.3a). Requires a Collocation with
        sigma_0 attached; see sigma0.attach_sigma0.

        Adds BOTH the interior equilibrium residual and the mechanical
        boundary conditions (`L_BC_mech`): base fixed, pit_floor and
        far_field_f1 rollers, and sigma_total . n = 0 on natural_ground,
        bench and cut_face. Every boundary Collocation must therefore carry
        sigma_0 as well as the interior one.

        The traction-free half is what D-3.2.5 refused to ship without. It is
        the condition on the cut face, which is the boundary the failure
        mechanism runs through, and without it the interior residuals can all
        be satisfied by a field that pushes arbitrarily hard on the free
        surface.
    """

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise KeyError(
                f"unknown loss weight(s) {sorted(unknown)}; "
                f"known: {sorted(DEFAULT_WEIGHTS)}"
            )
        w.update(weights)

    kc = KC_DEFAULT if include_feedback else KC_OFF
    pde, pde_parts = L_PDE(net, coll, mats, s, per_tag=per_term, kc=kc)
    icl, ic_parts = L_IC(net, *ic)
    bc, bc_parts = L_BC(net, bcs, mats, s, per_segment=per_term)
    iface, if_parts = L_interface(net, ifaces, mats, s, per_contact=per_term)

    contrib = {
        "contrib_pde": w["pde"] * pde,
        "contrib_ic": w["ic"] * icl,
        "contrib_bc": w["bc"] * bc,
        "contrib_interface": w["interface"] * iface,
    }
    mech_parts = {}
    if include_mechanics:
        mech, mech_parts = L_PDE_mech(net, coll, mats, s, per_tag=per_term)
        contrib["contrib_pde_mech"] = w["pde_mech"] * mech
        bcm, bcm_parts = L_BC_mech(net, bcs, mats, s, per_segment=per_term)
        contrib["contrib_bc_mech"] = w["bc_mech"] * bcm
        mech_parts = {**mech_parts, **bcm_parts}
    total = sum(contrib.values())

    parts = {**pde_parts, **mech_parts, **ic_parts, **bc_parts, **if_parts}
    parts.update({k: v.detach() for k, v in contrib.items()})
    parts["total"] = total.detach()
    return total, parts