import pytest
import torch

from src.materials import load_materials, theta, Se, K, C_star
from src.nondim import SCALES
from src import properties as P


@pytest.fixture(scope="module")
def mats():
    return load_materials()


def test_capacity_matches_autograd(mats):
    """C_star must equal the autograd derivative of theta, scaled."""
    for tag, m in mats.items():
        psi = torch.linspace(-50, -0.01, 200, dtype=torch.float64,
                             requires_grad=True)
        auto = torch.autograd.grad(theta(psi, m).sum(), psi)[0]
        auto = auto * SCALES.H_ref / SCALES.dtheta_ref
        assert torch.allclose(C_star(psi, m), auto, rtol=1e-8), tag


def test_capacity_positive(mats):
    """Water content rises as suction eases; C > 0 throughout."""
    psi = torch.linspace(-50, -0.01, 200, dtype=torch.float64)
    for tag, m in mats.items():
        assert (C_star(psi, m) > 0).all(), tag


def test_saturated_branch(mats):
    """psi >= 0: theta = theta_s, Se = 1, K = K_s."""
    psi = torch.full((5,), 1.0, dtype=torch.float64)
    for tag, m in mats.items():
        assert torch.allclose(theta(psi, m),
                              torch.full_like(psi, m.theta_s)), tag
        assert torch.allclose(Se(psi, m), torch.ones_like(psi)), tag
        assert torch.allclose(K(psi, m), torch.full_like(psi, m.K_s)), tag


def test_finite_at_zero(mats):
    """psi = 0 is the water table; values AND gradients must be finite there."""
    for tag, m in mats.items():
        for fn in (theta, Se, K, C_star):
            psi = torch.zeros(5, dtype=torch.float64, requires_grad=True)
            y = fn(psi, m)
            assert torch.isfinite(y).all(), f"{tag}: {fn.__name__} value"
            g = torch.autograd.grad(y.sum(), psi, allow_unused=True)[0]
            assert g is None or torch.isfinite(g).all(), \
                f"{tag}: {fn.__name__} grad"


def test_monotonic(mats):
    """Less suction -> more water, higher saturation, higher conductivity."""
    psi = torch.linspace(-50, -0.01, 500, dtype=torch.float64)
    for tag, m in mats.items():
        assert (torch.diff(theta(psi, m)) >= -1e-14).all(), f"{tag}: theta"
        assert (torch.diff(Se(psi, m)) >= -1e-14).all(), f"{tag}: Se"
        assert (torch.diff(K(psi, m)) >= -1e-14).all(), f"{tag}: K"


def test_relative_conductivity_bounded(mats):
    """K(psi) <= K_s everywhere: vg.K_r is RELATIVE, not absolute."""
    psi = torch.linspace(-50, -0.01, 200, dtype=torch.float64)
    for tag, m in mats.items():
        assert (K(psi, m) <= m.K_s * (1 + 1e-12)).all(), tag
        assert (K(psi, m) > 0).all(), tag


def test_porosity_tie_rule(mats):
    """properties.py adopts theta_s = 0.9 * n0 for all rock analogues."""
    for tag, m in mats.items():
        assert abs(m.theta_s - 0.9 * m.n0) < 1e-2, \
            f"{tag}: theta_s={m.theta_s} vs 0.9*n0={0.9 * m.n0}"


def test_units_and_provenance(mats):
    """Guard against unit-conversion regressions."""
    assert set(mats) == set(P.UNITS) == {"Mk", "Mk_d", "Tm"}
    for tag, m in mats.items():
        assert 1e6 < m.E < 1e11, f"{tag}: E={m.E:.3e} Pa, expected Pa not MPa"
        assert 1e-12 < m.K_s < 1e-2, f"{tag}: K_s={m.K_s:.3e} m/s"
        assert 0 < m.nu < 0.5, f"{tag}: nu={m.nu}"
        assert m.rho_sat > m.rho_dry, f"{tag}: rho_sat must exceed rho_dry"
        assert m.theta_s > m.theta_r, tag


def test_dtheta_ref_matches_mk(mats):
    """nondim.dtheta_ref is Mk's storage range; keep them in step."""
    mk = mats["Mk"]
    assert abs(SCALES.dtheta_ref - mk.dtheta) < 1e-6, \
        f"dtheta_ref={SCALES.dtheta_ref} vs Mk dtheta={mk.dtheta}"
