import pytest, torch
from src.config import BOUNDS, tiny
from src.loss import psi_of
from src.materials import load_materials
from src.model import PINN
from src.nondim import Scales
from src.residuals import richards_residual
from src.sampling import sample_interior

D1, D2 = 0.33, 0.377

@pytest.fixture
def probe():
    torch.manual_seed(0)
    c = sample_interior(200)
    m = load_materials()["Mk"]
    net = PINN(tiny(), BOUNDS)            # built ONCE
    def f(x, z, t):
        return psi_of(net(x, z, t))
    return f, c, m

def test_every_term_carries_the_same_dtheta_ref_power(probe):
    f, c, m = probe
    _, t1 = richards_residual(f, c.x, c.z, c.t, m, s=Scales(dtheta_ref=D1), return_terms=True)
    _, t2 = richards_residual(f, c.x, c.z, c.t, m, s=Scales(dtheta_ref=D2), return_terms=True)
    for key in ("storage", "diffusion", "gravity"):
        torch.testing.assert_close(D1 * t1[key], D2 * t2[key],
                                   rtol=1e-10, atol=0.0,
                                   msg=f"{key} is not homogeneous of degree -1 in dtheta_ref")

def test_residual_is_homogeneous_degree_minus_one(probe):
    f, c, m = probe
    R1 = richards_residual(f, c.x, c.z, c.t, m, s=Scales(dtheta_ref=D1))
    R2 = richards_residual(f, c.x, c.z, c.t, m, s=Scales(dtheta_ref=D2))
    torch.testing.assert_close(D1 * R1, D2 * R2, rtol=1e-10, atol=0.0)
    