"""Tests for the IC loss term.

Guards the failure modes that are silent rather than loud: a missing
H_ref division (factor-30 error), wrong output slicing, and float32
creeping into a path whose verification depends on float64.
"""

import torch

from src.config import BOUNDS, full
from src.loss import L_IC, L_PDE, ic_targets, psi_of, uv_of
from src.model import PINN
from src.nondim import SCALES
from src.sampling import sample_interior
from src.materials import load_materials


def test_slicing_shapes():
    """psi_of -> (N,1), uv_of -> (N,2); trailing dim preserved."""
    out = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    assert psi_of(out).shape == (4, 1)
    assert uv_of(out).shape == (4, 2)
    assert torch.equal(psi_of(out).squeeze(1), out[:, 0])


def test_psi0_is_nondimensional():
    """psi0* must be O(1), not O(100).

    Fails if the `/ H_ref` in ic_targets is dropped: psi0 is stored in
    metres over roughly -160..-3 m, so the dimensional version would
    exceed 100 in magnitude.
    """
    _, psi0 = ic_targets()
    assert psi0.abs().max() < 10.0, "psi0 looks dimensional -- H_ref division missing"
    assert psi0.abs().max() > 1.0, "psi0 suspiciously small -- double-divided?"
    assert (psi0 <= 0).all(), "unsaturated initial state should have psi <= 0"


def test_psi0_roundtrips_to_metres():
    """psi0* * H_ref recovers the stored metre range (-160..-3)."""
    _, psi0 = ic_targets()
    m = psi0 * SCALES.H_ref
    assert -161.0 < float(m.min()) < -155.0
    assert -4.0 < float(m.max()) < -2.0


def test_cache_invariants():
    """t* = 0 exactly; w is mean-normalised; 4000 samples."""
    coll, psi0 = ic_targets()
    assert float(coll.t.detach().abs().max()) == 0.0
    assert abs(float(coll.w.detach().mean()) - 1.0) < 1e-12
    assert psi0.shape == (4000, 1)


def test_dtype_is_float64():
    """float64 throughout -- the 1e-17 residual is meaningless in float32."""
    coll, psi0 = ic_targets()
    assert coll.x.dtype is torch.float64
    assert psi0.dtype is torch.float64


class _ExactNet:
    """Stub returning the IC target exactly, with zero displacements."""

    def __init__(self, psi0_star):
        self._psi0 = psi0_star

    def __call__(self, x, z, t):
        return torch.cat([self._psi0, torch.zeros_like(self._psi0),
                          torch.zeros_like(self._psi0)], dim=1)


def test_L_IC_vanishes_for_exact_solution():
    """A net reproducing psi0* with u = v = 0 gives L_IC = 0."""
    coll, psi0 = ic_targets()
    L, parts = L_IC(_ExactNet(psi0), coll, psi0)
    assert float(L) < 1e-28
    assert float(parts["ic_head"]) < 1e-28
    assert float(parts["ic_disp"]) == 0.0


def test_L_IC_detects_offset():
    """A constant offset d gives ic_head = d^2 under a mean-1 weighting."""
    coll, psi0 = ic_targets()
    d = 0.25
    L, parts = L_IC(_ExactNet(psi0 + d), coll, psi0)
    assert abs(float(parts["ic_head"]) - d ** 2) < 1e-12


def test_L_IC_runs_on_real_network():
    """End-to-end: finite, positive, float64, both components present."""
    coll, psi0 = ic_targets()
    net = PINN(full(), BOUNDS)
    L, parts = L_IC(net, coll, psi0)
    assert torch.isfinite(L)
    assert float(L.detach()) > 0.0
    assert L.dtype is torch.float64
    assert set(parts) == {"ic_head", "ic_disp"}

def test_L_PDE_gradients_reach_every_parameter():
    """Guards create_graph and the .detach() placement in L_PDE."""
    coll = sample_interior(200)
    net = PINN(full(), BOUNDS)
    L, _ = L_PDE(net, coll, load_materials())
    L.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.parameters())
    
def test_L_PDE_runs_on_real_network():
    """End-to-end: finite, positive, float64, per-tag components present."""
    coll = sample_interior(200)
    net = PINN(full(), BOUNDS)
    L, parts = L_PDE(net, coll, load_materials(), per_tag=True)
    assert torch.isfinite(L)
    assert float(L.detach()) > 0.0
    assert L.dtype is torch.float64
    assert set(parts) == {"pde_richards", "pde_richards_Mk",
                          "pde_richards_Mk_d", "pde_richards_Tm"}    