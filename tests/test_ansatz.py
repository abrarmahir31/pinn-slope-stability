"""Tests for `NearPhysical.mode` -- the Day 27 bounded-ansatz options.

The decision (DECISIONS.md D-A.1) is OPEN. These tests do not pick a winner.
They pin three things:

 1. the default is bit-identical to the shipped behaviour, so adding the option
    changed nothing;
 2. each bounded mode does what its name says -- psi stays negative, and the
    IC is or is not distorted by the documented amount;
 3. the finding that motivates the whole question -- that `unbounded` at
    eps_psi = 0.3 initialises a large fraction of the domain saturated -- is
    still true, because if it stops being true the option is unnecessary.

Second derivatives are checked explicitly: the Richards residual needs
div[K grad psi], so any ansatz that is not C^2 in z is unusable regardless of
how well it behaves at t = 0.
"""
import dataclasses

import pytest
import torch

from src.config import BOUNDS, tiny
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.sampling import sample_interior

H = SCALES.H_ref
Z_WT = NearPhysical.Z_WT


def _net(mode="unbounded", eps_psi=0.3, cap_k=100.0, seed=20250812, layers=8):
    cfg = dataclasses.replace(tiny(), n_layers=layers, n_neurons=64, seed=seed)
    return NearPhysical(PINN(cfg, BOUNDS), eps_psi=eps_psi,
                        mode=mode, cap_k=cap_k)


def _psi_m(net, coll):
    """psi in METRES on an interior collocation set."""
    return net(coll.x, coll.z, coll.t)[:, 0:1].detach().reshape(-1) * H


@pytest.fixture(scope="module")
def coll():
    return sample_interior(2000, 20250812)


def test_the_default_is_the_shipped_behaviour(coll):
    """Adding `mode` must not have changed anything. Bit-identical, not close.

    Every Step 3.4 weight, `PI_SEED_8X64`, and every checkpoint were produced
    under `psi* = psi0* + eps_psi * raw`. If the default drifts they are all
    silently invalidated.
    """
    net = _net()
    assert net.mode == "unbounded"

    raw = net.pinn(coll.x, coll.z, coll.t)[:, 0:1]
    psi0 = -(coll.z * SCALES.L_ref - Z_WT) / H
    expected = psi0 + net.eps_psi * raw
    got = net(coll.x, coll.z, coll.t)[:, 0:1]
    assert torch.equal(got, expected), (
        "the default ansatz no longer evaluates psi0* + eps_psi*raw")


@pytest.mark.parametrize("mode,cap_k", [("cap", 100.0), ("cap", 20.0),
                                        ("exp", None)])
def test_bounded_modes_never_saturate(coll, mode, cap_k):
    """psi < 0 everywhere, which is what the geometry says Section 5 is.

    Z_WT = 197 m sits below Z_BASE = 200 m, so there is no saturated region in
    the domain at any time under the current BC configuration.
    """
    kw = {} if cap_k is None else {"cap_k": cap_k}
    psi = _psi_m(_net(mode=mode, **kw), coll)
    assert (psi < 0).all(), (
        f"mode={mode}: {int((psi >= 0).sum())} of {len(psi)} points are at "
        f"psi >= 0, max {psi.max():.3f} m")


def test_unbounded_at_eps_psi_0_3_does_saturate(coll):
    """The finding the option exists for. Negative control for the test above.

    Measured Day 27 at 8x64 over three network seeds: 19.7% / 35.0% / 15.1% of
    points at psi >= 0, psi_max +54 to +185 m. If this ever stops being true,
    the bounded modes are solving a problem that no longer exists and D-A.1
    should be closed as unnecessary rather than decided.
    """
    psi = _psi_m(_net(mode="unbounded", eps_psi=0.3), coll)
    frac = float((psi >= 0).double().mean())
    assert frac > 0.05, (
        f"only {100 * frac:.1f}% of points initialise saturated under the "
        "unbounded ansatz; the Day 27 motivation for D-A.1 has changed")
    assert psi.max() > 0.0


def test_unbounded_at_eps_psi_3e_3_does_not(coll):
    """Why this went unnoticed for eleven days.

    At the pre-Day-25 prefactor a point at the floor needed raw > 6.1 to cross
    psi = 0 rather than raw > 0.06. The saturation problem was created by
    raising eps_psi, not by the unbounded output layer alone.
    """
    psi = _psi_m(_net(mode="unbounded", eps_psi=3e-3), coll)
    assert (psi < 0).all()
    assert psi.max() < -1.0


def test_the_ic_distortion_is_the_documented_amount():
    """At raw = 0 the ansatz must return psi_0, or it has moved the IC.

    `exp` and a sharp cap cost nothing. A soft cap does, and it costs most at
    psi_0 = -3 m -- the domain floor, and the only band where the Richards
    physics is live. That is the trade-off D-A.1 has to resolve, so it is
    pinned numerically rather than described.
    """
    z_floor = torch.tensor([[(Z_WT + 3.0) / SCALES.L_ref]], dtype=torch.float64)
    zero = torch.zeros_like(z_floor)

    def psi_at(mode, cap_k=100.0):
        return float(_net(mode=mode, cap_k=cap_k).psi_star(z_floor, zero)) * H

    assert psi_at("unbounded") == pytest.approx(-3.0, rel=1e-12)
    assert psi_at("exp") == pytest.approx(-3.0, rel=1e-12)
    assert psi_at("cap", 1000.0) == pytest.approx(-3.0, rel=1e-3)
    assert psi_at("cap", 100.0) == pytest.approx(-3.25, rel=1e-2)   # +8.3%
    assert psi_at("cap", 20.0) == pytest.approx(-7.35, rel=1e-2)    # +145%


@pytest.mark.parametrize("mode", ["unbounded", "cap", "exp"])
def test_every_mode_is_twice_differentiable_in_z(mode):
    """The Richards residual needs div[K* grad* psi*]. A mode that is not C^2
    in z is unusable no matter what it does to the seed."""
    net = _net(mode=mode, layers=2)
    x = torch.full((16, 1), 0.5, dtype=torch.float64, requires_grad=True)
    z = torch.full((16, 1), 1.5, dtype=torch.float64, requires_grad=True)
    t = torch.full((16, 1), 1.0, dtype=torch.float64, requires_grad=True)

    psi = net(x, z, t)[:, 0:1]
    d1, = torch.autograd.grad(psi.sum(), z, create_graph=True)
    d2, = torch.autograd.grad(d1.sum(), z, create_graph=True)
    assert torch.isfinite(d1).all(), f"{mode}: dpsi/dz is not finite"
    assert torch.isfinite(d2).all(), f"{mode}: d2psi/dz2 is not finite"
    assert d2.abs().sum() > 0, f"{mode}: second derivative is identically zero"


def test_bad_mode_and_cap_k_are_rejected():
    cfg = dataclasses.replace(tiny(), n_layers=2, n_neurons=64)
    with pytest.raises(ValueError, match="mode="):
        NearPhysical(PINN(cfg, BOUNDS), mode="bounded")
    with pytest.raises(ValueError, match="cap_k"):
        NearPhysical(PINN(cfg, BOUNDS), mode="cap", cap_k=0.0)


def test_summary_records_the_mode():
    """`vars(a)` goes into every checkpoint under "cfg", but `summary()` is
    what gets printed into run logs and pasted into notes. A run whose ansatz
    is invisible in its own log is unreproducible."""
    assert "mode=" not in _net().summary()               # default stays quiet
    assert "mode=cap k=100" in _net(mode="cap").summary()
    assert "mode=exp" in _net(mode="exp").summary()