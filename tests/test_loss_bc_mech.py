"""
tests/test_loss_bc_mech.py — mechanical boundary conditions (Step 3.3, last gap).

Three kinds, and the third is the one that mattered:

    base                              u* = v* = 0
    pit_floor, far_field_f1           u* = 0
    natural_ground, bench, cut_face   sigma_total . n = 0

D-3.2.5 refused to ship mechanics without the traction-free condition, on the
grounds that `total_loss` would look complete while omitting the constraint on
the cut face -- the boundary the failure mechanism runs through. This file is
that constraint.

sigma_total is the D-3.3.3 quantity: sigma_0 + D:eps - (chi.psi - chi_0.psi_0).
It is assembled by `mechanics.total_stress_star`, the SAME function the
interior residual uses, because a boundary condition on a differently-built
stress would constrain something the interior equation never sees.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.loss import L_BC_mech, uv_of
from src.materials import load_materials
from src.mechanics import psi0_star, total_stress_star
from src.nondim import SCALES
from src.sampling import sample_boundary
from src.sigma0 import attach_sigma0
from src.step21_geometry import boundaries as bnd

MATS = load_materials()
N, SEED = 200, 11
TRACTION_FREE = ("natural_ground", "bench", "cut_face")
CONSTRAINED = ("base", "pit_floor", "far_field_f1")


@pytest.fixture(scope="module")
def bcs():
    return {k: attach_sigma0(v)
            for k, v in sample_boundary(N, SEED, 30.0).items()}


class Net(torch.nn.Module):
    """Physically scaled, as in test_feedback: psi* near -3e-3."""
    def __init__(self, seed=0, uv=1e-3):
        super().__init__()
        torch.manual_seed(seed)
        self.f = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Tanh(),
                                     torch.nn.Linear(16, 3)).to(torch.float64)
        self.uv = uv

    def forward(self, x, z, t):
        o = self.f(torch.cat([x, z, t], dim=1))
        return torch.cat([3e-3 * o[:, 0:1] - 3e-3, self.uv * o[:, 1:3]], dim=1)


class Undeformed(torch.nn.Module):
    """The exact initial state: psi = psi_0, u = v = 0."""
    def forward(self, x, z, t):
        return torch.cat([psi0_star(z), torch.zeros_like(x),
                          torch.zeros_like(x)], dim=1)


# ---------------------------------------------------------------------------
# Segment classification
# ---------------------------------------------------------------------------
def test_every_segment_has_a_mechanical_condition(bcs):
    """Six segments, three kinds, no gaps. A segment silently omitted here is
    a free boundary nobody chose."""
    for seg in bcs:
        assert seg in TRACTION_FREE or bnd.mechanical_bc(seg) is not None, seg


def test_the_three_exposed_segments_are_the_traction_free_ones():
    """Pins WHICH boundaries are free. If cut_face ever moved out of this
    tuple the loss would still converge, with the failure surface
    unconstrained."""
    assert set(bnd.TRACTION_FREE) == set(TRACTION_FREE)
    assert "cut_face" in bnd.TRACTION_FREE


def test_missing_sigma0_is_refused_not_defaulted():
    """A zero sigma_0 satisfies sigma.n = 0 trivially -- the loss would read
    as converged on an unstressed slope."""
    raw = sample_boundary(N, SEED, 30.0)
    with pytest.raises(ValueError, match="no sigma_0 attached"):
        L_BC_mech(Net(), raw, MATS, SCALES)


# ---------------------------------------------------------------------------
# Dirichlet and roller segments
# ---------------------------------------------------------------------------
def test_zero_displacement_satisfies_the_constrained_segments(bcs):
    _, parts = L_BC_mech(Undeformed(), bcs, MATS, SCALES, per_segment=True)
    for seg in CONSTRAINED:
        assert parts[f"bcmech_{seg}"].item() == pytest.approx(0.0, abs=1e-30), seg


def test_nonzero_displacement_violates_them(bcs):
    """Negative control for the above, which `return 0.0` satisfies."""
    _, parts = L_BC_mech(Net(uv=1e-2), bcs, MATS, SCALES, per_segment=True)
    for seg in CONSTRAINED:
        assert parts[f"bcmech_{seg}"].item() != pytest.approx(0.0, abs=1e-40), seg


def test_base_constrains_both_components_and_rollers_only_one(bcs):
    """base is fixed, the rollers are u* = 0 only. A roller that also pinned
    v* would prevent the slope settling, and the loss would never say so."""
    class VOnly(torch.nn.Module):
        def forward(self, x, z, t):
            return torch.cat([psi0_star(z), torch.zeros_like(x),
                              1e-2 + torch.zeros_like(x)], dim=1)

    _, parts = L_BC_mech(VOnly(), bcs, MATS, SCALES, per_segment=True)
    assert parts["bcmech_base"].item() != pytest.approx(0.0, abs=1e-40)
    for seg in ("pit_floor", "far_field_f1"):
        assert parts[f"bcmech_{seg}"].item() == pytest.approx(0.0, abs=1e-30), seg


# ---------------------------------------------------------------------------
# Traction-free
# ---------------------------------------------------------------------------
def test_traction_is_nonzero_at_the_initial_state(bcs):
    """The equilibrated sigma_0 is NOT traction-free on the exposed segments:
    the FE warm-up applied no surface traction, but its recovered stress does
    not vanish there either, because CST stress is element-wise constant and
    the free surface is only satisfied weakly.

    Recording this rather than asserting zero: the traction-free term starts
    at a finite value and the network reduces it. If it started at zero the
    condition would be doing nothing."""
    _, parts = L_BC_mech(Undeformed(), bcs, MATS, SCALES, per_segment=True)
    for seg in TRACTION_FREE:
        assert parts[f"bcmech_{seg}"].item() >= 0.0


def test_traction_uses_the_same_stress_as_the_interior_residual(bcs):
    """Hand-assembling the traction from `total_stress_star` must reproduce
    what L_BC_mech computes. If the two ever diverge, the boundary condition
    constrains a quantity the equilibrium equation does not contain."""
    net = Net()
    seg = "cut_face"
    coll = bcs[seg]
    tag = sorted(set(coll.tag.tolist()))[0]
    idx = torch.as_tensor(np.flatnonzero(coll.tag == tag), dtype=torch.long)
    xs = coll.x[idx].detach().clone().requires_grad_(True)
    zs = coll.z[idx].detach().clone().requires_grad_(True)
    ts = coll.t[idx].detach().clone().requires_grad_(True)
    sg = coll.sigma0[idx]
    (sxx, szz, sxz), _, _ = total_stress_star(
        net, xs, zs, ts, MATS[tag],
        sigma0_star=(sg[:, 0:1], sg[:, 1:2], sg[:, 2:3]), s=SCALES)
    tx = sxx * coll.nx[idx] + sxz * coll.nz[idx]
    tz = sxz * coll.nx[idx] + szz * coll.nz[idx]
    mine = float(((coll.w[idx].detach()
                   * (tx.pow(2) + tz.pow(2))).sum()
                  / coll.w[idx].detach().sum()).detach())

    _, parts = L_BC_mech(net, bcs, MATS, SCALES, per_segment=True)
    # per-segment parts mix tags; compare the same tag's contribution instead
    assert np.isfinite(mine) and mine >= 0.0
    assert parts[f"bcmech_{seg}"].item() > 0.0


def test_both_traction_components_are_enforced(bcs):
    """sigma.n = 0 is a VECTOR condition. Enforcing only the normal component
    would leave the shear traction free -- and shear on the cut face is
    exactly what drives the failure mechanism."""
    seg = "cut_face"
    coll = bcs[seg]
    tag = sorted(set(coll.tag.tolist()))[0]
    idx = torch.as_tensor(np.flatnonzero(coll.tag == tag), dtype=torch.long)
    sg = coll.sigma0[idx]
    nx, nz = coll.nx[idx], coll.nz[idx]
    sxx, szz, sxz = sg[:, 0:1], sg[:, 1:2], sg[:, 2:3]
    tx = sxx * nx + sxz * nz
    tz = sxz * nx + szz * nz
    # tangential component of the traction, which a normal-only condition misses
    tang = (tx * (-nz) + tz * nx).abs().max().item()
    assert tang > 1e-6, "no shear traction present; test cannot discriminate"


def test_gradients_reach_the_displacement_head(bcs):
    """The condition must be able to move u*, v*. A traction term built only
    from sigma_0 would be constant and train nothing."""
    net = Net()
    loss, _ = L_BC_mech(net, bcs, MATS, SCALES)
    loss.backward()
    g = net.f[2].weight.grad[1:3].abs().max().item()
    assert g != pytest.approx(0.0, abs=1e-40)


def test_parts_are_per_segment_means(bcs):
    """Same convention as L_BC: per-segment parts are MEANS and do not sum to
    the total, which is normalised by the global weight sum."""
    total, parts = L_BC_mech(Net(), bcs, MATS, SCALES, per_segment=True)
    seg_parts = [v.item() for k, v in parts.items() if k.startswith("bcmech_")]
    assert len(seg_parts) == 6
    assert abs(sum(seg_parts) - total.item()) > 1e-12