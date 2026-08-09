"""dtheta_ref homogeneity: residual is degree -1, L_PDE is degree -2.

Regression guard for ff5c4be, where the C_star closure in
swcc_from_material scaled its argument with the injected s but let
_m.C_star fall back to the SCALES singleton.
"""
import pytest, torch

from src.config import BOUNDS, tiny
from src.loss import psi_of
from src.materials import load_materials
from src.model import PINN
from src.nondim import Scales
from src.residuals import richards_residual
from src.sampling import sample_interior

D1, D2 = 0.33, 0.377
SCALED = ("storage", "diffusion", "gravity")   # degree -1
INVARIANT = ("K_star", "dpsi_dz")              # degree  0


@pytest.fixture
def probe():
    torch.manual_seed(0)
    c = sample_interior(200)
    m = load_materials()["Mk"]
    net = PINN(tiny(), BOUNDS)          # built ONCE, outside the sweep

    def f(x, z, t):
        return psi_of(net(x, z, t))

    def run(d):
        return richards_residual(f, c.x, c.z, c.t, m,
                                 s=Scales(dtheta_ref=d), return_terms=True)
    return run


def test_terms_are_degree_minus_one(probe):
    _, t1 = probe(D1)
    _, t2 = probe(D2)
    for key in SCALED:
        torch.testing.assert_close(
            D1 * t1[key], D2 * t2[key], rtol=1e-10, atol=0.0,
            msg=f"{key} is not homogeneous of degree -1 in dtheta_ref")


def test_scale_free_terms_do_not_move(probe):
    _, t1 = probe(D1)
    _, t2 = probe(D2)
    for key in INVARIANT:
        torch.testing.assert_close(t1[key], t2[key], rtol=1e-12, atol=0.0,
                                   msg=f"{key} should not depend on dtheta_ref")


def test_residual_is_degree_minus_one(probe):
    R1, _ = probe(D1)
    R2, _ = probe(D2)
    torch.testing.assert_close(D1 * R1, D2 * R2, rtol=1e-10, atol=0.0)