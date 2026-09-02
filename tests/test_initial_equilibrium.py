"""
tests/test_initial_equilibrium.py — the state the whole coupled model rests on.

ONE claim, checked several ways: at t = 0, with u* = v* = 0 and psi* = psi_0*,
against the equilibrated sigma_0, the mechanical residual is ZERO.

That is not a convenience. It is the definition of "equilibrated": the FE
gravity solve (Step 3.3a) produced a sigma_0 that balances the body force
exactly, so the undeformed initial state is a solution of the equilibrium
equation and the network has nothing to correct at t = 0. If the residual is
nonzero there, training begins by fitting an error that is not physics, and
every displacement it produces is a response to that error.

WHY THIS FILE EXISTS
--------------------
`test_coupling.py::test_zero_displacement_and_equilibrated_sigma0_gives_zero_residual`
made the same claim with `bishop=False`. It passed, and it was vacuous: the
term that broke the property was the one it switched off. The residual with
bishop=True was 0.604 (Mk) and 0.579 (Mk_d), about 20% of Pi_M_body, for
weeks -- see docs/open_items.md and DECISIONS.md D-3.3.3.

Two separate causes, so two separate groups of tests below.
"""

from __future__ import annotations

import inspect
import pytest
import torch

from src.materials import load_materials
from src.mechanics import chi_of, mechanical_residual, psi0_star
from src.nondim import SCALES
from src.sampling import sample_interior
from src.sigma0 import attach_sigma0

MATS = load_materials()
N, SEED = 1200, 7


@pytest.fixture(scope="module")
def coll():
    return attach_sigma0(sample_interior(N, SEED))


def _undeformed(x, z, t):
    """The exact initial state: psi = psi_0(z), no displacement.

    psi_0 is built from z INSIDE the field, not passed in as a constant. The
    pore term enters the residual through div, so what matters is its spatial
    gradient; a detached psi_0 has none and behaves exactly like the absolute
    form. That mistake made an earlier version of this test pass a formulation
    that was still wrong.
    """
    return torch.cat([psi0_star(z), torch.zeros_like(x), torch.zeros_like(x)],
                     dim=1)


def _residual(coll, tag, **kw):
    m = torch.as_tensor(coll.tag == tag)
    sg = coll.sigma0[m]
    return mechanical_residual(
        _undeformed, coll.x[m], coll.z[m], coll.t[m], MATS[tag],
        sigma0_star=(sg[:, 0:1], sg[:, 1:2], sg[:, 2:3]),
        rho0_ratio=coll.rho0[m], **kw)


# ---------------------------------------------------------------------------
# psi_0 itself
# ---------------------------------------------------------------------------
def test_psi0_star_matches_the_initial_condition_module():
    """`psi0_star` is an analytic restatement of `initial.psi_initial` so that
    it can be differentiated. If the two ever disagree, the mechanical residual
    is equilibrated against a different initial state than the hydraulic one.
    """
    from src.step21_geometry.initial import psi_initial
    z_phys = torch.linspace(200.0, 360.0, 40, dtype=torch.float64).reshape(-1, 1)
    z_star = z_phys / SCALES.L_ref
    got = psi0_star(z_star) * SCALES.H_ref
    want = torch.as_tensor(psi_initial(0.0, z_phys.numpy()), dtype=torch.float64)
    assert torch.allclose(got, want, rtol=1e-12, atol=1e-9)


def test_the_domain_is_entirely_unsaturated_at_t0():
    """Z_WT = 197 < Z_BASE = 200. Every point is in suction, so chi_0 < 1
    everywhere and the initial pore term is NOT simply p_0 = 0. If this ever
    fails the geometry has moved and D-3.3.3's premise needs rechecking."""
    z_star = torch.linspace(200.0, 360.0, 50, dtype=torch.float64).reshape(-1, 1) / SCALES.L_ref
    assert (psi0_star(z_star) < 0).all()


def test_chi0_is_far_from_both_zero_and_one():
    """Negative control on the above. If chi_0 were ~0 the increment would
    equal the absolute form and this bug would never have mattered; if it were
    ~1 the pore term would be plain Terzaghi. Measured over the real elevation
    range (psi_0 from -3.5 to -163 m):

        chi_0(Mk, Mk_d)   0.414 .. 0.835
        chi_0(Tm)         0.003 .. 0.141

    The marls sit in the middle of the Bishop range, which is why the fix moved
    their residual by 0.6 and Tm's by only 1e-6 -- Tm has desaturated so far
    that its pore term is nearly absent (Finding 3)."""
    z_star = torch.linspace(200.5, 360.0, 40, dtype=torch.float64).reshape(-1, 1) / SCALES.L_ref
    chi0 = chi_of(psi0_star(z_star), MATS["Mk"])
    assert 0.3 < chi0.min().item()
    assert chi0.max().item() < 0.95
    chi0_tm = chi_of(psi0_star(z_star), MATS["Tm"])
    assert chi0_tm.max().item() < 0.25


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag", ["Mk", "Mk_d", "Tm"])
def test_initial_state_is_in_equilibrium_with_bishop_on(coll, tag):
    """THE test. Was 0.604 / 0.579 / 1.06e-06 before D-3.3.3."""
    res_x, res_z = _residual(coll, tag, bishop=True)
    assert res_x.abs().max().item() == pytest.approx(0.0, abs=1e-10), f"{tag} x"
    assert res_z.abs().max().item() == pytest.approx(0.0, abs=1e-10), f"{tag} z"


@pytest.mark.parametrize("tag", ["Mk", "Mk_d", "Tm"])
def test_initial_state_is_in_equilibrium_with_bishop_off(coll, tag):
    """The old, weaker version. Kept because it isolates the sigma_0 / rho_0
    bookkeeping from the pore term: if BOTH fail the FE cache is wrong, if only
    the bishop=True one fails the pore term is."""
    res_x, res_z = _residual(coll, tag, bishop=False)
    assert res_x.abs().max().item() == pytest.approx(0.0, abs=1e-10)
    assert res_z.abs().max().item() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Negative controls -- without these the tests above are satisfied by
# `mechanical_residual` returning zeros.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag", ["Mk", "Mk_d"])
def test_perturbing_the_suction_breaks_equilibrium(coll, tag):
    """A psi that is not psi_0 must produce a nonzero residual, otherwise the
    pore term is not reaching the equilibrium equation at all and the coupling
    is decorative."""
    m = torch.as_tensor(coll.tag == tag)
    sg = coll.sigma0[m]

    def wetter(x, z, t):
        return torch.cat([psi0_star(z) + 0.05, torch.zeros_like(x),
                          torch.zeros_like(x)], dim=1)

    _, res_z = mechanical_residual(
        wetter, coll.x[m], coll.z[m], coll.t[m], MATS[tag],
        sigma0_star=(sg[:, 0:1], sg[:, 1:2], sg[:, 2:3]),
        rho0_ratio=coll.rho0[m], bishop=True)
    assert res_z.abs().max().item() != pytest.approx(0.0, abs=1e-40)


@pytest.mark.parametrize("tag", ["Mk", "Mk_d"])
def test_displacing_the_solid_breaks_equilibrium(coll, tag):
    m = torch.as_tensor(coll.tag == tag)
    sg = coll.sigma0[m]

    def pushed(x, z, t):
        return torch.cat([psi0_star(z), 1e-3 * x * x, torch.zeros_like(x)], dim=1)

    res_x, _ = mechanical_residual(
        pushed, coll.x[m], coll.z[m], coll.t[m], MATS[tag],
        sigma0_star=(sg[:, 0:1], sg[:, 1:2], sg[:, 2:3]),
        rho0_ratio=coll.rho0[m], bishop=True)
    assert res_x.abs().max().item() != pytest.approx(0.0, abs=1e-40)


def test_a_detached_psi0_would_not_fix_it(coll):
    """Guards the subtlety that cost a debugging round. The increment
    chi.psi - chi_0.psi_0 is identically zero in VALUE at t = 0, so it is
    tempting to build psi_0 as a constant. Detached, its gradient vanishes and
    the term behaves exactly like the absolute form. This reproduces that and
    asserts it is NOT what the code does."""
    tag = "Mk"
    m = torch.as_tensor(coll.tag == tag)
    z = coll.z[m]
    live = psi0_star(z)
    dead = psi0_star(z).detach()
    assert live.requires_grad
    assert not dead.requires_grad
    (g,) = torch.autograd.grad(live.sum(), z, create_graph=True)
    assert g.abs().min().item() != pytest.approx(0.0, abs=1e-40)


def cfg_eps_psi():
    """The production eps_psi, read off NearPhysical's signature so this test
    tracks model.py rather than duplicating the constant."""
    from src.model import NearPhysical
    return inspect.signature(NearPhysical.__init__).parameters["eps_psi"].default


def test_nearphysical_preserves_interface_continuity_at_t0():
    """The wrapper, not psi0_star. eps_psi scales the network's contribution,
    so raising it perturbs the init away from exact flux continuity. This is
    the test that would have caught the 3e-3 suppression, and the one that
    bounds how far eps_psi can go."""
    from src.config import bounds_from_geometry, full
    from src.loss import L_interface
    from src.model import PINN, NearPhysical
    from src.sampling import sample_interfaces

    ifaces = sample_interfaces(400, SEED)

    cfg = full()
    cfg.device, cfg.dtype = "cpu", "float64"
    bounds = bounds_from_geometry(t_max=30.0, s=SCALES)

    # The analytic limit: eps_psi = 0 leaves psi = psi0_star exactly, which
    # has uniform total head and therefore zero Darcy flux on both sides of
    # every contact. test_loss_interface.py pins that this vanishes.
    exact = NearPhysical(PINN(cfg, bounds), eps_psi=0.0)
    assert float(L_interface(exact, ifaces, MATS)[0].detach()) == pytest.approx(0.0, abs=1e-20)

    # At the production value the network's contribution is live, so the jump
    # is no longer identically zero. TOLERANCE IS A MODELLING CHOICE: measured
    # 5.85e-06 at eps_psi = 0.3 (2 Sep, seed 7, N=400), against Pi_M_body =
    # 3.0019, i.e. ~2e-6 of the scale the mechanical residual is normalised to.
    # The 1e-4 bound leaves ~17x headroom for seed variation. Raising eps_psi
    # past the point where this fails means the ansatz no longer starts from a
    # flux-continuous state, and D-W.4's "interface is inactive at t = 0"
    # premise no longer holds.
    prod = NearPhysical(PINN(cfg, bounds), eps_psi=cfg_eps_psi())
    assert float(L_interface(prod, ifaces, MATS)[0].detach()) < 1e-4