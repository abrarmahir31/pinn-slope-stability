import pytest
import torch

from src.materials import load_materials, theta, K, C
from src.nondim import SCALES


@pytest.fixture(scope="module")
def mats():
    with pytest.warns(UserWarning):
        return load_materials()


def test_capacity_matches_autograd(mats):
    for tag, m in mats.items():
        psi = torch.linspace(-50, -0.01, 200, dtype=torch.float64, requires_grad=True)
        auto = torch.autograd.grad(theta(psi, m).sum(), psi)[0]
        auto = auto * SCALES.H_ref / SCALES.dtheta_ref
        assert torch.allclose(C(psi, m), auto, rtol=1e-10), tag
