"""
tests/test_feedback.py — Step 3.3b, the hydraulic half of two-way coupling.

Kozeny-Carman enters K* inside the Richards residual, so the Richards residual
becomes a function of u*, v*. That dependency is the coupling; most of this
file exists to prove it is really there when it should be and really absent
when it should not.

The load-bearing pair:
  * `test_k_factor_none_and_ones_are_identical` — the one-way arm of the
    Step 6.2 ablation must reproduce the pre-Step-3.3 loss exactly, not
    approximately.
  * `test_feedback_creates_a_gradient_path_to_the_displacements` — and the
    two-way arm must actually differ. Either alone is satisfiable by a stub.
"""

from __future__ import annotations

import pytest
import torch

from src.coupling import KC_DEFAULT, KC_OFF, KCConfig
from src.derivatives import as_inputs
from src.loss import L_PDE, ic_targets, total_loss
from src.materials import load_materials
from src.nondim import SCALES
from src.residuals import richards_residual, richards_residual_expanded
from src.sampling import (sample_boundary, sample_interfaces, sample_interior)
from src.sigma0 import attach_sigma0

MATS = load_materials()
MK = MATS["Mk"]
N, SEED = 300, 20250812
KC_ALL = KCConfig(enabled=True)


class Net(torch.nn.Module):
    """Three outputs, PHYSICALLY SCALED. Two separate scales, deliberately.

    psi* is held near -3e-3, i.e. about half a metre of suction. An unscaled
    random net outputs psi* of order 1, which is 165 m of suction, where the
    van Genuchten K_r is numerically zero -- the diffusion and gravity terms
    then sit 12 orders below storage and NOTHING done to K* is visible. Two
    tests here failed that way before this class was scaled, and they were
    right to: a test net that puts the model in a regime it will never occupy
    cannot detect a change to that regime.

    u*, v* are left larger, because eps_v is what Kozeny-Carman consumes and
    it is 100x smaller than eps_v* (the U_ref/L_ref factor).
    """
    def __init__(self, seed=0, psi_scale=3e-3, uv_scale=10.0):
        super().__init__()
        torch.manual_seed(seed)
        self.f = torch.nn.Sequential(
            torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 3)
        ).to(torch.float64)
        self.psi_scale, self.uv_scale = psi_scale, uv_scale

    def forward(self, x, z, t):
        o = self.f(torch.cat([x, z, t], dim=1))
        return torch.cat([self.psi_scale * o[:, 0:1] - 3e-3,
                          self.uv_scale * o[:, 1:3]], dim=1)


def _pts(n=24, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.float64) * 1.4
    z = torch.rand(n, 1, generator=g, dtype=torch.float64) * 0.9 + 1.2
    t = torch.zeros(n, 1, dtype=torch.float64)
    return as_inputs(x, z, t)


# ---------------------------------------------------------------------------
# richards_residual: the k_factor argument itself
# ---------------------------------------------------------------------------
def test_k_factor_none_and_ones_are_identical():
    """EXACT, not approximate. The whole ablation rests on it."""
    x, z, t = _pts()
    f = lambda a, b, c: -0.4 + 0.2 * b + 0.05 * a
    R_none = richards_residual(f, x, z, t, MK, SCALES)
    R_ones = richards_residual(f, x, z, t, MK, SCALES,
                               k_factor=torch.ones_like(x))
    assert torch.equal(R_none, R_ones)


def test_k_factor_changes_the_residual():
    """Negative control for the test above, which `k_factor = ignore` passes."""
    x, z, t = _pts()
    f = lambda a, b, c: -0.4 + 0.2 * b + 0.05 * a
    R1 = richards_residual(f, x, z, t, MK, SCALES)
    R2 = richards_residual(f, x, z, t, MK, SCALES,
                           k_factor=torch.full_like(x, 2.0))
    assert (R1 - R2).abs().max().item() != pytest.approx(0.0, abs=1e-40)


def test_a_constant_k_factor_scales_only_the_flux_terms():
    """Storage is C* dpsi*/dt* and has no K in it, so a uniform doubling of K
    must leave it alone while doubling diffusion and gravity. Pins WHERE the
    factor enters, which a global rescale of R would also satisfy."""
    x, z, t = _pts()
    f = lambda a, b, c: -0.4 + 0.2 * b + 0.05 * a + 0.1 * c
    _, T1 = richards_residual(f, x, z, t, MK, SCALES, return_terms=True)
    _, T2 = richards_residual(f, x, z, t, MK, SCALES, return_terms=True,
                              k_factor=torch.full_like(x, 2.0))
    assert torch.allclose(T1["storage"], T2["storage"], rtol=1e-12)
    assert torch.allclose(2 * T1["diffusion"], T2["diffusion"], rtol=1e-10)
    assert torch.allclose(2 * T1["gravity"], T2["gravity"], rtol=1e-10)


def test_a_varying_k_factor_is_differentiated_not_treated_as_constant():
    """The factor multiplies K BEFORE div and d/dz are taken, so a factor with
    a spatial gradient contributes through the product rule. If it were applied
    afterwards, this would equal the constant-factor case at the same mean."""
    x, z, t = _pts()
    f = lambda a, b, c: -0.4 + 0.2 * b + 0.05 * a
    varying = 1.0 + 0.3 * (z - z.mean())
    R_var = richards_residual(f, x, z, t, MK, SCALES, k_factor=varying)
    R_con = richards_residual(f, x, z, t, MK, SCALES,
                              k_factor=torch.full_like(x, float(varying.mean().detach())))
    assert (R_var - R_con).abs().max().item() != pytest.approx(0.0, abs=1e-40)


def test_wrong_shaped_k_factor_is_rejected():
    """A (1,1) factor broadcasts happily and would apply one point's feedback
    to every point."""
    x, z, t = _pts()
    f = lambda a, b, c: -0.4 + 0.2 * b
    with pytest.raises(ValueError, match="k_factor has shape"):
        richards_residual(f, x, z, t, MK, SCALES,
                          k_factor=torch.ones(1, 1, dtype=torch.float64))


def test_expanded_form_refuses_a_k_factor():
    """Its analytic chain rule assumes K = K(psi) only. Ignoring the factor
    would make the cross-check pass while comparing two different equations."""
    x, z, t = _pts()
    f = lambda a, b, c: -0.4 + 0.2 * b
    with pytest.raises(NotImplementedError, match="eps_v"):
        richards_residual_expanded(f, x, z, t, MK, SCALES,
                                   k_factor=torch.ones_like(x))


# ---------------------------------------------------------------------------
# L_PDE: the factor is built from the network's displacements
# ---------------------------------------------------------------------------
def test_feedback_off_reproduces_the_one_way_loss_exactly():
    net = Net()
    coll = sample_interior(N, SEED)
    a, _ = L_PDE(net, coll, MATS, SCALES, kc=KC_OFF)
    b, _ = L_PDE(net, coll, MATS, SCALES)          # default is KC_OFF
    assert float(a.detach()) == float(b.detach())


def test_feedback_on_changes_the_hydraulic_loss():
    net = Net()                          # displacements worth having
    coll = sample_interior(N, SEED)
    off, _ = L_PDE(net, coll, MATS, SCALES, kc=KC_OFF)
    on, _ = L_PDE(net, coll, MATS, SCALES, kc=KC_DEFAULT)
    assert abs(float(on.detach()) - float(off.detach())) > 1e-30


def test_zero_displacement_makes_feedback_on_equal_feedback_off():
    """The control that catches a flag which arrives and is never read: with
    eps_v = 0 the Kozeny-Carman factor is exactly 1, so the two must agree
    even though one takes the multiplier path."""
    class Undeformed(Net):
        def forward(self, x, z, t):
            psi = super().forward(x, z, t)[:, 0:1]
            return torch.cat([psi, torch.zeros_like(x), torch.zeros_like(x)], 1)

    net = Undeformed()
    coll = sample_interior(N, SEED)
    off, _ = L_PDE(net, coll, MATS, SCALES, kc=KC_OFF)
    on, _ = L_PDE(net, coll, MATS, SCALES, kc=KC_DEFAULT)
    assert float(on.detach()) == pytest.approx(float(off.detach()), rel=1e-12)


def test_feedback_creates_a_gradient_path_to_the_displacements():
    """THE test. Two-way coupling means the HYDRAULIC residual depends on the
    displacement head. If this passes with feedback off, the graph is wired
    somewhere it should not be; if it fails with feedback on, the factor was
    detached and the coupling is one-way in disguise."""
    def head_grad(kc):
        net = Net()
        coll = sample_interior(N, SEED)
        loss, _ = L_PDE(net, coll, MATS, SCALES, kc=kc)
        loss.backward()
        # the last layer's rows 1 and 2 are the u*, v* outputs
        return net.f[2].weight.grad[1:3].abs().max().item()

    assert head_grad(KC_OFF) == pytest.approx(0.0, abs=1e-40)
    assert head_grad(KC_DEFAULT) != pytest.approx(0.0, abs=1e-40)


def test_Tm_is_excluded_from_the_feedback(caplog):
    """D-3.3.2 at the loss level, not just in the config object. Tm is 87.3% of
    the domain, so an accidental inclusion would dominate and still look
    plausible."""
    net = Net()
    coll = sample_interior(N, SEED)
    default, _ = L_PDE(net, coll, MATS, SCALES, kc=KC_DEFAULT)
    everywhere, _ = L_PDE(net, coll, MATS, SCALES, kc=KC_ALL)
    assert abs(float(default.detach()) - float(everywhere.detach())) > 1e-30


# ---------------------------------------------------------------------------
# total_loss: the 2x2 ablation
# ---------------------------------------------------------------------------
def _bundle():
    return dict(coll=attach_sigma0(sample_interior(N, SEED)),
                bcs=sample_boundary(N, SEED, 30.0), ic=ic_targets(),
                ifaces=sample_interfaces(N, SEED), mats=MATS)


def test_the_four_ablation_arms_are_all_distinct():
    """Step 6.2 needs feedback and mechanics to be independent switches, so
    that 'the effect of two-way coupling' can be separated from 'the effect of
    adding a mechanical residual at all'."""
    net = Net()
    b = _bundle()
    vals = {}
    for fb in (False, True):
        for mech in (False, True):
            v, _ = total_loss(net, b["coll"], b["bcs"], b["ic"], b["ifaces"],
                              b["mats"], include_feedback=fb,
                              include_mechanics=mech)
            vals[(fb, mech)] = float(v.detach())
    assert vals[(False, False)] != vals[(True, False)]
    assert vals[(False, False)] != vals[(False, True)]
    assert len(set(vals.values())) == 4


def test_both_switches_off_is_the_pre_step_33_loss():
    net = Net()
    b = _bundle()
    a, _ = total_loss(net, b["coll"], b["bcs"], b["ic"], b["ifaces"], b["mats"])
    c, _ = total_loss(net, b["coll"], b["bcs"], b["ic"], b["ifaces"], b["mats"],
                      include_feedback=False, include_mechanics=False)
    assert float(a.detach()) == float(c.detach())