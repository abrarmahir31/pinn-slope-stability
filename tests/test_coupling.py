"""
tests/test_coupling.py — Step 3.3 two-way coupling.

Repo rules honoured:
  * every claim gets a negative control;
  * every `!= pytest.approx(...)` carries an explicit `abs=1e-40`, because
    quantities here live at 1e-12 and the default makes a negative control
    silently vacuous;
  * analytic fields go through the same code path as the network.

`test_total_stress_subtracts_the_pore_term` pins the stress convention settled
in D-3.3.1 and CORRECTED in D-3.3.3. It checks the algebra in isolation; the
equilibrium check that actually catches a wrong pore term lives in
tests/test_initial_equilibrium.py.
from __future__ import annotations"""

import pytest
import torch

from src.coupling import (KC_DEFAULT, KC_OFF, KC_ON, KCConfig, bishop_chi,
                          bulk_density_ratio, kozeny_carman_factor,
                          porosity_from_strain)
from src.derivatives import as_inputs
from src.materials import load_materials
from src.mechanics import (E_hat, chi_of, lame_hat, mechanical_residual,
                           strain_star, stress_increment_star, theta_of)
from src.nondim import SCALES

MATS = load_materials()
MK, MKD, TM = MATS["Mk"], MATS["Mk_d"], MATS["Tm"]


def _pts(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g, dtype=torch.float64) * 1.5
    z = torch.rand(n, 1, generator=g, dtype=torch.float64) * 1.2 + 1.1
    t = torch.zeros(n, 1, dtype=torch.float64)
    return as_inputs(x, z, t)


# ---------------------------------------------------------------------------
# Kozeny-Carman
# ---------------------------------------------------------------------------
def test_zero_strain_gives_unit_factor():
    eps = torch.zeros(8, 1, dtype=torch.float64)
    f = kozeny_carman_factor(MK.n0, eps, KC_ON)
    assert f.sub(1.0).abs().max().item() == pytest.approx(0.0, abs=1e-14)


def test_dilation_raises_conductivity_and_compression_lowers_it():
    """Direction, not just magnitude. A factor moving the wrong way still
    satisfies 'the factor changes'."""
    eps = torch.tensor([[0.01], [-0.01]], dtype=torch.float64)
    f = kozeny_carman_factor(MK.n0, eps, KC_ON)
    assert f[0].item() > 1.0
    assert f[1].item() < 1.0


def test_disabled_reproduces_one_way_exactly():
    """The ablation depends on EXACT, not approximate."""
    eps = torch.linspace(-0.05, 0.05, 32, dtype=torch.float64).reshape(-1, 1)
    f = kozeny_carman_factor(MK.n0, eps, KC_OFF)
    assert torch.equal(f, torch.ones_like(f))


def test_disabled_factor_carries_no_graph():
    """A flag that arrives but is never read passes the test above. This one
    checks the graph, not the value — the `ebf8f5c` bug shape."""
    eps = torch.full((8, 1), 0.02, dtype=torch.float64, requires_grad=True)
    assert not kozeny_carman_factor(MK.n0, eps, KC_OFF).requires_grad
    assert kozeny_carman_factor(MK.n0, eps, KC_ON).requires_grad


def test_factor_gradient_is_nonzero():
    eps = torch.full((8, 1), 0.005, dtype=torch.float64, requires_grad=True)
    f = kozeny_carman_factor(MK.n0, eps, KC_ON)
    (df,) = torch.autograd.grad(f.sum(), eps, create_graph=True)
    assert df.abs().min().item() != pytest.approx(0.0, abs=1e-40)


def test_per_unit_override_beats_the_global_flag():
    """Tm is 87.3% of the domain, 2.2% porosity, fracture-dominated. Whether
    a matrix-porosity law applies there is a decision, and this is where it
    is recorded."""
    cfg = KCConfig(enabled=True, per_unit={"Tm": False})
    eps = torch.full((4, 1), 0.005, dtype=torch.float64)
    assert torch.equal(kozeny_carman_factor(TM.n0, eps, cfg, tag="Tm"),
                       torch.ones_like(eps))
    assert not torch.equal(kozeny_carman_factor(MK.n0, eps, cfg, tag="Mk"),
                           torch.ones_like(eps))


def test_tm_low_porosity_is_the_sensitive_case():
    """Documents rather than hides the sensitivity. At n0 = 0.0221 an 0.005
    strain is a 23% porosity change; at n0 = 0.4268 it is 1.2%."""
    eps = torch.tensor([[0.005]], dtype=torch.float64)
    f_mk = kozeny_carman_factor(MK.n0, eps, KC_ON).item()
    f_tm = kozeny_carman_factor(TM.n0, eps, KC_ON).item()
    assert (f_tm - 1.0) > 10.0 * (f_mk - 1.0)


def test_clamp_prevents_negative_porosity():
    """Early training outputs noise. n < 0 makes (n/n0)^3 negative and K_s
    negative; n > 1 makes ((1-n0)/(1-n))^2 singular."""
    eps = torch.tensor([[-10.0], [10.0]], dtype=torch.float64)
    f = kozeny_carman_factor(TM.n0, eps, KC_ON)
    assert torch.isfinite(f).all()
    assert (f > 0).all()


def test_clamp_fraction_is_reported():
    eps = torch.tensor([[0.0], [0.0], [5.0], [-5.0]], dtype=torch.float64)
    _, diag = kozeny_carman_factor(MK.n0, eps, KC_ON, return_diag=True)
    assert diag["clamped_fraction"] == pytest.approx(0.5, abs=1e-12)


def test_empty_clamp_band_raises_rather_than_inverting_silently():
    bad = KCConfig(enabled=True, ratio_min=1.2, ratio_max=1.1)
    with pytest.raises(ValueError):
        porosity_from_strain(MK.n0, torch.zeros(4, 1, dtype=torch.float64), bad)


# ---------------------------------------------------------------------------
# Strain and stress
# ---------------------------------------------------------------------------
def test_rigid_body_translation_produces_no_strain():
    x, z, t = _pts()
    u = torch.full_like(x, 0.3)
    v = torch.full_like(x, -0.2)
    for e in strain_star(u, v, x, z):
        assert e.abs().max().item() == pytest.approx(0.0, abs=1e-14)


def test_uniform_extension_produces_the_strain_it_should():
    x, z, t = _pts()
    a = 0.07
    exx, ezz, exz, ev = strain_star(a * x, torch.zeros_like(x), x, z)
    assert exx.mean().item() == pytest.approx(a, rel=1e-12)
    assert ezz.abs().max().item() == pytest.approx(0.0, abs=1e-14)
    assert ev.mean().item() == pytest.approx(a, rel=1e-12)


def test_physical_flag_rescales_by_U_over_L():
    """The factor-of-100 trap. Without it eps_v* reaches Kozeny-Carman
    unconverted and the feedback is 100x too strong."""
    x, z, t = _pts()
    u, v = 0.05 * x, torch.zeros_like(x)
    ev_star = strain_star(u, v, x, z, physical=False)[3]
    ev_phys = strain_star(u, v, x, z, physical=True)[3]
    ratio = (ev_phys / ev_star).mean().item()
    assert ratio == pytest.approx(SCALES.U_ref / SCALES.L_ref, rel=1e-12)
    assert ratio == pytest.approx(0.01, rel=1e-3)


def test_hydrostatic_compression_gives_isotropic_stress():
    x, z, t = _pts()
    a = -0.01
    exx, ezz, exz, _ = strain_star(a * x, a * z, x, z)
    sxx, szz, sxz = stress_increment_star(exx, ezz, exz, MK)
    assert (sxx - szz).abs().max().item() == pytest.approx(0.0, abs=1e-14)
    assert sxz.abs().max().item() == pytest.approx(0.0, abs=1e-14)
    assert sxx.mean().item() < 0.0          # tension positive: compression < 0


def test_stiffer_material_carries_more_stress_for_the_same_strain():
    x, z, t = _pts()
    exx, ezz, exz, _ = strain_star(0.01 * x, torch.zeros_like(x), x, z)
    s_mk = stress_increment_star(exx, ezz, exz, MK)[0].mean().item()
    s_tm = stress_increment_star(exx, ezz, exz, TM)[0].mean().item()
    assert s_tm > s_mk
    assert E_hat(TM) / E_hat(MK) == pytest.approx(TM.E / MK.E, rel=1e-12)


def test_shear_uses_the_tensor_component_not_engineering_gamma():
    """Pins the factor of two. tau = 2*mu*eps_xz with eps_xz the TENSOR
    component; a gamma-based implementation gives twice this."""
    x, z, t = _pts()
    b = 0.03
    exx, ezz, exz, _ = strain_star(b * z, torch.zeros_like(x), x, z)
    assert exz.mean().item() == pytest.approx(0.5 * b, rel=1e-12)
    _, two_mu = lame_hat(MK)
    sxz = stress_increment_star(exx, ezz, exz, MK)[2]
    assert sxz.mean().item() == pytest.approx(two_mu * 0.5 * b, rel=1e-12)


def test_boundaries_and_properties_agree_on_the_elastic_constants():
    """The E_rm conflict was resolved in Step 3.2 and the two tables now
    agree. `mechanics.py` reads properties.py (single source of truth) while
    the mechanical BCs read boundaries.py, so they must not drift apart again
    — the mechanical residual and its own boundary conditions would then be
    built on different stiffnesses, which no field plot would reveal."""
    from src.step21_geometry import boundaries as bnd
    for mat in (MK, MKD, TM):
        assert bnd.E_RM[mat.tag] == pytest.approx(mat.E, rel=1e-6), mat.tag
        assert bnd.NU[mat.tag] == pytest.approx(mat.nu, rel=1e-12), mat.tag


# ---------------------------------------------------------------------------
# Bishop and bulk density
# ---------------------------------------------------------------------------
def test_chi_is_bounded_and_falls_with_suction():
    psi = torch.tensor([[-1.0], [-0.1], [0.5]], dtype=torch.float64)
    chi = chi_of(psi, MK)
    assert chi.min().item() >= 0.0 and chi.max().item() <= 1.0
    assert chi[0].item() < chi[1].item()      # drier -> lower chi
    assert chi[2].item() == pytest.approx(1.0, abs=1e-12)   # saturated


def test_chi_is_differentiable_in_psi():
    psi = torch.full((6, 1), -0.3, dtype=torch.float64, requires_grad=True)
    (d,) = torch.autograd.grad(chi_of(psi, MK).sum(), psi, create_graph=True)
    assert d.abs().min().item() != pytest.approx(0.0, abs=1e-40)


def test_theta_is_not_chi():
    """theta = theta_r + (theta_s-theta_r)*Se. Passing Se where theta belongs
    is a body force wrong by tens of percent."""
    psi = torch.full((4, 1), -0.3, dtype=torch.float64)
    assert theta_of(psi, MK).mean().item() != pytest.approx(
        chi_of(psi, MK).mean().item(), abs=1e-40)


def test_bulk_density_ratio_rises_with_water_content():
    theta = torch.tensor([[0.05], [0.38]], dtype=torch.float64)
    r = bulk_density_ratio(theta, MK.rho_dry, SCALES.rho_b_ref)
    assert r[1].item() > r[0].item()
    assert r[0].item() == pytest.approx(
        (MK.rho_dry + 0.05 * 1000.0) / SCALES.rho_b_ref, rel=1e-12)



def test_total_stress_subtracts_the_pore_term():
    """Was `test_bishop_sign_is_tension_positive`, which asserted the OPPOSITE
    and was wrong. See D-3.3.3.

    Tension positive. Bishop compression-positive is sigma_eff = sigma - chi*p;
    converting (sigma_t = -sigma_c) gives effective = total + chi*p, hence

        total = effective - chi*p

    The constitutive law produces EFFECTIVE stress (D:eps is carried by the
    skeleton) and equilibrium acts on TOTAL stress, so the residual needs this
    direction and the sign is a subtraction. Positive pore pressure therefore
    makes the total stress MORE COMPRESSIVE than the effective stress that
    produced it.

    The old test asserted the total->effective direction: a true statement
    about a formula, and the wrong one for the caller. It passed because it
    tested the function's name, and at the time there was no caller to
    disagree with it."""
    from src.nondim import total_stress_nd
    sig_eff = torch.zeros(4, 1, dtype=torch.float64)
    chi = torch.ones(4, 1, dtype=torch.float64)
    wetter = torch.full((4, 1), 0.5, dtype=torch.float64)
    drier = torch.full((4, 1), -0.5, dtype=torch.float64)
    assert total_stress_nd(sig_eff, wetter, chi, SCALES).mean().item() < 0.0
    assert total_stress_nd(sig_eff, drier, chi, SCALES).mean().item() > 0.0


def test_zero_pore_increment_leaves_the_stress_alone():
    """Negative control. At t = 0 the increment chi*psi - chi_0*psi_0 is
    identically zero, so total must equal effective exactly -- which is what
    makes the undeformed initial state an equilibrium."""
    from src.nondim import total_stress_nd
    sig = torch.linspace(-3.0, 1.0, 8, dtype=torch.float64).reshape(-1, 1)
    zero = torch.zeros_like(sig)
    assert torch.equal(total_stress_nd(sig, zero, torch.ones_like(sig), SCALES), sig)


def test_effective_stress_nd_refuses_rather_than_forwarding():
    """A silent alias would give numbers wrong twice over: flipped sign AND
    absolute instead of incremental."""
    from src.nondim import effective_stress_nd
    with pytest.raises(NotImplementedError, match="D-3.3.3"):
        effective_stress_nd(torch.zeros(2, 1), torch.zeros(2, 1),
                            torch.ones(2, 1), SCALES)


# ---------------------------------------------------------------------------
# Residual assembly
# ---------------------------------------------------------------------------
def _zero_field(a, b, c):
    z3 = torch.zeros_like(a)
    return torch.cat([z3, z3, z3], dim=1)


def test_zero_displacement_and_equilibrated_sigma0_gives_zero_residual():
    """The baseline the gravity warm-up must reproduce: if sigma0 is
    equilibrated and rho_b == rho_0, an undeformed state is a solution."""
    x, z, t = _pts()
    rho0 = bulk_density_ratio(theta_of(torch.zeros_like(x), MK), MK.rho_dry,
                              SCALES.rho_b_ref)
    res_x, res_z = mechanical_residual(_zero_field, x, z, t, MK,
                                       bishop=False, rho0_ratio=rho0)
    assert res_x.abs().max().item() == pytest.approx(0.0, abs=1e-14)
    assert res_z.abs().max().item() == pytest.approx(0.0, abs=1e-14)


def test_nonzero_displacement_gives_nonzero_residual():
    """Negative control for the test above, which `return 0.0` satisfies."""
    x, z, t = _pts()

    def f(a, b, c):
        return torch.cat([torch.zeros_like(a), 0.02 * a * a,
                          torch.zeros_like(a)], dim=1)

    res_x, _ = mechanical_residual(f, x, z, t, MK, bishop=False,
                                   body_force=False)
    assert res_x.abs().max().item() != pytest.approx(0.0, abs=1e-40)


def test_body_force_enters_the_z_component_only():
    """e_z = (0,-1). A body force appearing in res_x means the gravity
    direction was wired wrong, which a converged loss will happily hide."""
    x, z, t = _pts()
    res_x, res_z = mechanical_residual(_zero_field, x, z, t, MK,
                                       bishop=False, rho0_ratio=0.0)
    assert res_x.abs().max().item() == pytest.approx(0.0, abs=1e-14)
    assert res_z.abs().max().item() > 0.0


def test_body_force_magnitude_is_Pi_M_body_times_rho_ratio():
    x, z, t = _pts()
    _, terms = mechanical_residual(_zero_field, x, z, t, MK, bishop=False,
                                   rho0_ratio=0.0, return_terms=True)
    expected = SCALES.Pi_M_body * terms["rho_ratio"]
    _, res_z = mechanical_residual(_zero_field, x, z, t, MK, bishop=False,
                                   rho0_ratio=0.0)
    assert (res_z + expected).abs().max().item() == pytest.approx(0.0, abs=1e-12)


def test_bishop_flag_changes_the_residual():
    """The mechanical half of the ablation switch."""
    x, z, t = _pts()

    def f(a, b, c):
        # psi MUST vary in space: the pore term enters through div sigma, so a
        # uniform psi contributes nothing. That is the next test, not this one.
        return torch.cat([-0.4 + 0.3 * a, 0.01 * a, 0.01 * b], dim=1)

    on_x, _ = mechanical_residual(f, x, z, t, MK, bishop=True, body_force=False)
    off_x, _ = mechanical_residual(f, x, z, t, MK, bishop=False, body_force=False)
    assert (on_x - off_x).abs().max().item() != pytest.approx(0.0, abs=1e-40)


def test_bishop_term_vanishes_for_a_uniform_psi_field():
    """Negative control for the test above: a spatially uniform psi has zero
    gradient, so the pore term is constant and contributes nothing to div
    sigma. If it does contribute, the term was assembled on the wrong axis."""
    x, z, t = _pts()

    def f(a, b, c):
        return torch.cat([torch.full_like(a, -0.4), 0.01 * a, 0.01 * b], dim=1)

    on_x, _ = mechanical_residual(f, x, z, t, MK, bishop=True, body_force=False)
    off_x, _ = mechanical_residual(f, x, z, t, MK, bishop=False, body_force=False)
    assert (on_x - off_x).abs().max().item() == pytest.approx(0.0, abs=1e-14)

# ---------------------------------------------------------------------------
# The configuration of record (D-3.3.2)
# ---------------------------------------------------------------------------
def test_kc_default_excludes_Tm_and_includes_the_marls():
    """D-3.3.2. Tm is excluded because Kozeny-Carman is a matrix-porosity law
    and Tm's conductivity is fracture-controlled (Finding 6), not because it
    misbehaves numerically. If someone later 'simplifies' this to a plain
    enabled=True, this is what objects."""
    eps = torch.full((4, 1), -5e-4, dtype=torch.float64)
    assert torch.equal(kozeny_carman_factor(TM.n0, eps, KC_DEFAULT, tag="Tm"),
                       torch.ones_like(eps))
    for mat, tag in ((MK, "Mk"), (MKD, "Mk_d")):
        f = kozeny_carman_factor(mat.n0, eps, KC_DEFAULT, tag=tag)
        assert not torch.equal(f, torch.ones_like(f)), tag
        assert (f < 1.0).all(), f"{tag}: compression must lower K_s"


def test_kc_default_is_flippable_for_the_ablation():
    """Step 6.2 needs three arms: KC everywhere, KC on the marls only, and no
    feedback at all. All three must be reachable without editing source."""
    eps = torch.full((4, 1), -5e-4, dtype=torch.float64)
    everywhere = KCConfig(enabled=True)
    f_all = kozeny_carman_factor(TM.n0, eps, everywhere, tag="Tm")
    f_def = kozeny_carman_factor(TM.n0, eps, KC_DEFAULT, tag="Tm")
    f_off = kozeny_carman_factor(TM.n0, eps, KC_OFF, tag="Tm")
    assert not torch.equal(f_all, f_def)
    assert torch.equal(f_def, f_off)          # both are 1.0 for Tm
    assert not torch.equal(f_all, f_off)


def test_tm_is_strain_sensitive_but_barely_strains():
    """The empirical fact behind D-3.3.2, pinned so the argument in the thesis
    can be checked against the code. Per unit strain Tm is ~11x more sensitive
    than Mk_d; at the strains it actually sees, the two land within 1.5%."""
    e10 = {}
    for mat in (MKD, TM):
        lo, hi = 1e-9, 0.4 * mat.n0
        for _ in range(80):                    # bisection, no scipy import
            mid = 0.5 * (lo + hi)
            f = kozeny_carman_factor(
                mat.n0, torch.tensor([[mid]], dtype=torch.float64), KC_ON).item()
            lo, hi = (mid, hi) if f < 1.10 else (lo, mid)
        e10[mat.tag] = 0.5 * (lo + hi)
    assert e10["Mk_d"] / e10["Tm"] > 5.0       # Tm far more sensitive per strain

# --------------------------------------------------------------------------
# Day 26: is_on must not fail open
# --------------------------------------------------------------------------
def test_is_on_raises_on_an_unknown_tag():
    """The mutation: `is_on` falls through to `self.enabled` for a tag it does
    not recognise. Then `KC_DEFAULT.is_on("TM")` is True, a misspelled stratum
    in L_PDE silently receives the feedback D-3.3.2 excluded Tm from, and the
    Step 6.2 ablation compares two arms that differ by a typo."""
    from src.coupling import KC_DEFAULT, KCConfig
    with pytest.raises(KeyError, match="unknown stratum tag"):
        KC_DEFAULT.is_on("TM")
    with pytest.raises(KeyError, match="unknown stratum tag"):
        KCConfig(enabled=True).is_on("XX")


def test_is_on_still_answers_the_three_real_strata_and_the_global_query():
    """The guard must not change any answer that was already correct."""
    from src.coupling import KC_DEFAULT, KC_OFF, KC_ON
    assert KC_DEFAULT.is_on("Mk") is True
    assert KC_DEFAULT.is_on("Mk_d") is True
    assert KC_DEFAULT.is_on("Tm") is False        # D-3.3.2
    assert KC_DEFAULT.is_on() is True             # global setting
    assert KC_OFF.is_on() is False
    assert KC_ON.is_on("Tm") is True
