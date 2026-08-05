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


def test_saturated_branch(mats):
    """At psi >= 0 the medium is saturated: theta = theta_s, K = K_s, C = 0."""
    for tag, m in mats.items():
        psi = torch.full((5,), 1.0, dtype=torch.float64)
        assert torch.allclose(theta(psi, m), torch.full_like(psi, m.theta_s)), tag
        assert torch.allclose(K(psi, m), torch.full_like(psi, m.K_s)), tag
        assert torch.allclose(C(psi, m), torch.zeros_like(psi)), tag


def test_finite_at_zero(mats):
    """psi = 0 sits on the water table and is hit constantly during training.

    Values AND gradients must be finite there, or the wetting front produces
    NaN gradients that are near-impossible to trace back to source.
    """
    for tag, m in mats.items():
        for fn in (theta, K, C):
            psi = torch.zeros(5, dtype=torch.float64, requires_grad=True)
            y = fn(psi, m)
            assert torch.isfinite(y).all(), f"{tag}: {fn.__name__} value"
            g = torch.autograd.grad(y.sum(), psi, allow_unused=True)[0]
            # None is legitimate: the saturated branch is constant in psi.
            assert g is None or torch.isfinite(g).all(), f"{tag}: {fn.__name__} grad"


def test_monotonic(mats):
    """Less suction -> more water and higher conductivity."""
    for tag, m in mats.items():
        psi = torch.linspace(-50, -0.01, 500, dtype=torch.float64)
        assert (torch.diff(theta(psi, m)) >= 0).all(), f"{tag}: theta"
        assert (torch.diff(K(psi, m)) >= 0).all(), f"{tag}: K"


def test_loader_sanity(mats):
    assert set(mats) == {"Tm", "Mk", "Mk_d"}
    assert mats["Tm"].provisional is True
    assert mats["Mk"].provisional is False
    assert mats["Mk_d"].provisional is False
    # E must be Pa, not MPa: 478 MPa -> 4.78e8
    for tag, m in mats.items():
        assert 1e7 < m.E < 1e11, f"{tag}: E = {m.E:.3e} Pa, unit conversion suspect"
    # theta_s was overridden to n0 in load_materials
    assert mats["Mk"].theta_s == mats["Mk"].n0
    assert mats["Mk"].theta_s > mats["Mk"].theta_r
