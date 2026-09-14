"""Pins `residuals.flux_fraction` and the claims the activity_floor rewrite
rests on.

Same philosophy as `test_ds4_term_scales.py`: the material-ceiling tests
RECOMPUTE from `properties.py` and `nondim.py` rather than reading
`docs/ds4_term_scales.json`, because the failure mode being guarded against is
a silent parameter drift that leaves the decision reading as measured when it
is no longer true.

If `test_the_marls_cannot_reach_any_plausible_threshold` fails, DO NOT relax
the threshold in the test. It means the marls' attainable flux fraction has
moved, which changes whether a global cut on p is admissible at all, and that
is a decision to reread rather than a number to patch.

Nothing here imports `scripts/` -- no test in this repo does, and `scripts/`
has no `__init__.py`.
"""
import math

import pytest
import torch

from src.materials import load_materials
from src.nondim import SCALES
from src.residuals import flux_fraction, swcc_from_material

PSI_M = (-0.01, -0.1, -1.0, -3.0, -10.0, -30.0, -80.0, -161.0)

#: Any threshold a person would plausibly write down for "the fluxes are
#: live". Both readings considered on Day 28 -- the note's 0.5 and the
#: revision's 0.05 -- sit at or above this.
PLAUSIBLE_THRESHOLD = 0.05


def terms(storage, diffusion, gravity):
    """A terms dict shaped like `richards_residual(..., return_terms=True)`."""
    f = torch.as_tensor
    return {"storage": f(storage, dtype=torch.float64),
            "diffusion": f(diffusion, dtype=torch.float64),
            "gravity": f(gravity, dtype=torch.float64)}


def attainable_ceiling(tag, s=SCALES):
    """max p over the D-S.4 psi sweep, from src. Mirrors
    `scripts/ds4_term_scales.ratios`."""
    swcc = swcc_from_material(load_materials()[tag], s)
    best = 0.0
    for psi_m in PSI_M:
        ps = torch.tensor([psi_m / s.H_ref], dtype=torch.float64,
                          requires_grad=True)
        K = swcc.K_star(ps)
        C = swcc.C_star(ps)
        dK, = torch.autograd.grad(K.sum(), ps)
        D = (s.Pi_R_diff * K / C).item()
        G = (s.Pi_R_grav * dK / C).abs().item()
        best = max(best, max(D, G) / (1.0 + D + G))
    return best


# -- the definition ---------------------------------------------------------
def test_it_is_bounded_in_the_unit_interval():
    torch.manual_seed(0)
    t = terms(torch.rand(500) * 1e3, torch.rand(500) * 1e-9,
              torch.rand(500) * 1e3)
    p = flux_fraction(t)
    assert float(p.min()) >= 0.0
    assert float(p.max()) <= 1.0


def test_collapsed_fluxes_read_as_inert():
    """The degenerate case the rule exists to catch: C* dpsi*/dt* = 0."""
    p = flux_fraction(terms([1.0], [1e-12], [1e-13]))
    assert float(p[0]) < 1e-11


def test_storage_balanced_against_one_flux_reads_one_half():
    """What a solved transient Richards equation looks like."""
    p = flux_fraction(terms([1.0], [1.0], [0.0]))
    assert float(p[0]) == pytest.approx(0.5)


def test_vanished_storage_reads_as_quasi_steady():
    p = flux_fraction(terms([1e-14], [1.0], [0.0]))
    assert float(p[0]) > 0.999


def test_the_larger_flux_wins_regardless_of_which_it_is():
    a = flux_fraction(terms([1.0], [2.0], [0.0]))
    b = flux_fraction(terms([1.0], [0.0], [2.0]))
    assert float(a[0]) == pytest.approx(float(b[0]))


def test_sign_does_not_matter():
    a = flux_fraction(terms([1.0], [-2.0], [0.5]))
    b = flux_fraction(terms([-1.0], [2.0], [-0.5]))
    assert float(a[0]) == pytest.approx(float(b[0]))


def test_an_identically_zero_point_is_inert_not_nan():
    """0/0 carries no regime information. NaN would poison any reduction
    downstream silently, which is the class of bug this whole signal exists
    to replace."""
    p = flux_fraction(terms([0.0], [0.0], [0.0]))
    assert not math.isnan(float(p[0]))
    assert float(p[0]) == 0.0


def test_it_does_not_enter_the_graph():
    s = torch.ones(4, dtype=torch.float64, requires_grad=True)
    d = torch.ones(4, dtype=torch.float64, requires_grad=True)
    g = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    p = flux_fraction({"storage": s, "diffusion": d, "gravity": g})
    assert not p.requires_grad


def test_a_terms_dict_without_return_terms_is_refused():
    with pytest.raises(KeyError, match="richards_residual"):
        flux_fraction({"storage": torch.ones(3)})


def test_it_returns_one_value_per_point():
    p = flux_fraction(terms(torch.ones(7, 1), torch.ones(7, 1),
                            torch.zeros(7, 1)))
    assert p.shape == (7,)


# -- the claims the rewrite rests on ----------------------------------------
@pytest.mark.parametrize("tag", ["Mk", "Mk_d"])
def test_the_marls_cannot_reach_any_plausible_threshold(tag):
    """THE LOAD-BEARING ONE. The attainable ceiling on p in the marls is a
    MATERIAL property, not a training state -- so `freeze when p < 0.05` is
    an unconditional disable of `pde_richards` in 12.7% of the domain, at any
    saturation, however well the network is solving. Recomputed, not read."""
    assert attainable_ceiling(tag) < PLAUSIBLE_THRESHOLD / 10.0


def test_tm_and_the_marls_differ_by_orders_so_no_global_cut_exists():
    """3.3 orders on the measured parameters. If this ever falls below ~2,
    a single global threshold becomes arguable and the per-material
    normalisation could be dropped -- which is a decision, not a test fix."""
    marl = max(attainable_ceiling(t) for t in ("Mk", "Mk_d"))
    assert math.log10(attainable_ceiling("Tm") / marl) > 2.0


def test_tm_alone_can_reach_the_live_regime():
    """The converse: scoping the rule to Tm is viable because Tm's ceiling is
    O(1). If this fails, the signal is useless everywhere and the rule needs
    rethinking rather than retuning."""
    assert attainable_ceiling("Tm") > 0.5


def test_the_storage_coefficient_vanishes_at_saturation():
    """THE MECHANISM behind the `unbounded` confound, and a type change in the
    equation rather than a large number. van Genuchten C* = dtheta*/dpsi* is
    identically zero at psi = 0, so at a saturated point the storage term is
    not small -- it is absent, and Richards degenerates to a steady-state flux
    balance. Any psi >= 0 point therefore reads p = 1.000 exactly."""
    swcc = swcc_from_material(load_materials()["Tm"], SCALES)
    ps = torch.zeros(1, dtype=torch.float64)
    assert float(swcc.C_star(ps)) == 0.0


def test_approaching_saturation_reads_as_maximally_live():
    """The same confound just below saturation, where C* is nonzero and the
    ratio is finite. Pinned so it cannot quietly disappear from the script's
    psi >= 0 warning."""
    s = SCALES
    swcc = swcc_from_material(load_materials()["Tm"], s)
    ps = torch.tensor([-1e-6 / s.H_ref], dtype=torch.float64,
                      requires_grad=True)
    K = swcc.K_star(ps)
    C = swcc.C_star(ps)
    dK, = torch.autograd.grad(K.sum(), ps)
    D = (s.Pi_R_diff * K / C).item()
    G = (s.Pi_R_grav * dK / C).abs().item()
    assert max(D, G) / (1.0 + D + G) > 0.9


# -- the methodological claim -----------------------------------------------
def test_identical_term_medians_can_hide_a_forty_point_difference():
    """Why `flux_fraction` returns a tensor rather than a float.

    NOT because the medians disagree -- p built from the per-stratum medians
    tracks the median of p to 1.0-1.9x on a matched field, and an earlier
    claim of ~3 orders here was an artifact of comparing a trained checkpoint
    against an untrained network. The medians are fine.

    What the aggregate cannot express is the SPREAD. These two fields have
    identical medians in all three terms, so any rule reading aggregated
    ratios sees one number for both -- while one is wholly inert and 40% of
    the other is at p = 0.5. The freeze decision is about how much of the
    domain has degenerated, which a central value cannot carry."""
    n = 1000
    dead = terms(torch.ones(n), torch.full((n,), 1e-9), torch.zeros(n))

    storage = torch.ones(n, dtype=torch.float64)
    storage[: 4 * n // 10] = 1e-9                    # a live 40%
    mixed = terms(storage, torch.full((n,), 1e-9), torch.zeros(n))

    for k in ("storage", "diffusion", "gravity"):
        assert float(dead[k].median()) == pytest.approx(float(mixed[k].median()))

    f_dead = float((flux_fraction(dead) > 1e-3).double().mean())
    f_mixed = float((flux_fraction(mixed) > 1e-3).double().mean())
    assert f_dead == pytest.approx(0.0)
    assert f_mixed == pytest.approx(0.4)


def test_a_fraction_survives_a_tail_that_moves_a_quantile():
    """Why the reduction is `frac(p > p_ref)` and not a percentile. Contaminate
    1% of points and the 99th percentile moves by orders while the live
    fraction barely registers -- which is the Day-26 seed-variance failure
    mode (469.9x on one draw) refusing to reappear one layer up."""
    n = 1000
    base = terms(torch.ones(n), torch.full((n,), 1e-6), torch.zeros(n))
    p_base = flux_fraction(base)

    d = torch.full((n,), 1e-6, dtype=torch.float64)
    d[: n // 100] = 1e6                 # 1% extreme
    p_spike = flux_fraction(terms(torch.ones(n), d, torch.zeros(n)))

    q_ratio = (float(torch.quantile(p_spike, 0.99))
               / float(torch.quantile(p_base, 0.99)))
    f_base = float((p_base > 1e-3).double().mean())
    f_spike = float((p_spike > 1e-3).double().mean())

    assert q_ratio > 1e3                        # the quantile is dragged
    assert abs(f_spike - f_base) <= 0.011       # the fraction is not