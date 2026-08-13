"""Tests for `total_loss` (Step 3.2 deliverable).

Each term is already pinned by its own module. This tests only assembly:
weighting, the contrib/parts distinction, and that mechanics is absent rather
than half-present.

The central claim is `total == sum(contrib_*)`. Per-term diagnostic parts are
MEANS and do not sum to anything; `contrib_*` are weighted contributions and
do. Conflating them is the mistake this file exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.nondim import SCALES as S
from src.loss import total_loss, DEFAULT_WEIGHTS, _psi_field, ic_targets
from src.sampling import sample_interior, sample_boundary, sample_interfaces
from src.materials import load_materials

N = 300
SEED = 20250812
MATS = load_materials()
DTYPE = torch.float64
TERMS = ("pde", "ic", "bc", "interface")
MECH_TERMS = TERMS + ("pde_mech",)


@pytest.fixture(scope="module")
def bundle():
    """Everything total_loss needs, sampled once."""
    return dict(
        coll=sample_interior(N, SEED),
        bcs=sample_boundary(N, SEED, 30.0),
        ic=ic_targets(),
        ifaces=sample_interfaces(N, SEED),
        mats=MATS,
    )


@pytest.fixture(scope="module")
def net():
    return TinyNet()


class TinyNet(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 3),
        ).to(DTYPE)

    def forward(self, x, z, t):
        return self.net(torch.cat([x, z, t], dim=1))


def call(net, bundle, **kw):
    return total_loss(net, bundle["coll"], bundle["bcs"], bundle["ic"],
                      bundle["ifaces"], bundle["mats"], **kw)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_returns_scalar_and_parts(net, bundle):
    total, parts = call(net, bundle)
    assert total.ndim == 0
    assert "total" in parts
    assert float(parts["total"]) == pytest.approx(float(total.detach()),
                                                  rel=1e-12)


def test_every_term_reports_a_contribution(net, bundle):
    _, parts = call(net, bundle)
    for term in TERMS:
        assert f"contrib_{term}" in parts, f"contrib_{term} missing"


def test_all_parts_are_detached(net, bundle):
    _, parts = call(net, bundle, per_term=True)
    for k, v in parts.items():
        assert not torch.as_tensor(v).requires_grad, f"parts[{k}] not detached"


def test_contributions_sum_to_the_total(net, bundle):
    """The identity that makes the parts dict readable. Per-term diagnostic
    parts are means and do NOT sum to anything; contrib_* are weighted
    contributions and do."""
    total, parts = call(net, bundle)
    summed = sum(float(parts[f"contrib_{t}"]) for t in TERMS)
    assert summed == pytest.approx(float(total.detach()), rel=1e-12)


def test_diagnostic_parts_do_not_sum_to_the_total(net, bundle):
    """Negative control for the test above: if these summed too, the two kinds
    of entry would be indistinguishable and someone would log the wrong one."""
    total, parts = call(net, bundle, per_term=True)
    diagnostic = {k: v for k, v in parts.items()
                  if not k.startswith("contrib_") and k != "total"}
    summed = sum(float(v) for v in diagnostic.values())
    assert summed != pytest.approx(float(total.detach()), rel=1e-6, abs=1e-40)


def test_per_term_is_opt_in(net, bundle):
    """Default off, matching per_tag / per_segment / per_contact."""
    _, lean = call(net, bundle)
    _, full = call(net, bundle, per_term=True)
    assert len(full) > len(lean), "per_term=True added no breakdown"
    for k in lean:
        assert k in full, f"{k} vanished when per_term=True"


# ---------------------------------------------------------------------------
# Weighting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("term", TERMS)
def test_zero_weight_removes_a_term_entirely(net, bundle, term):
    _, parts = call(net, bundle, weights={term: 0.0})
    assert float(parts[f"contrib_{term}"]) == 0.0
    for other in TERMS:
        if other != term:
            assert float(parts[f"contrib_{other}"]) != 0.0, (
                f"zeroing {term} also silenced {other}"
            )


@pytest.mark.parametrize("term", TERMS)
def test_contribution_is_linear_in_its_weight(net, bundle, term):
    _, one = call(net, bundle, weights={term: 1.0})
    _, three = call(net, bundle, weights={term: 3.0})
    assert float(three[f"contrib_{term}"]) == pytest.approx(
        3.0 * float(one[f"contrib_{term}"]), rel=1e-9, abs=1e-40
    )


def test_weights_default_to_one(net, bundle):
    a, _ = call(net, bundle)
    b, _ = call(net, bundle, weights=dict(DEFAULT_WEIGHTS))
    assert float(b.detach()) == pytest.approx(float(a.detach()), rel=1e-12)
    assert set(DEFAULT_WEIGHTS) == set(MECH_TERMS)


def test_unknown_weight_key_is_rejected(net, bundle):
    """A typo in a weight name must not silently train with the default."""
    with pytest.raises(KeyError, match="unknown loss weight"):
        call(net, bundle, weights={"mechanics": 1.0})


# ---------------------------------------------------------------------------
# Mechanics is absent, not half-present
# ---------------------------------------------------------------------------
def test_include_mechanics_without_sigma0_refuses(net, bundle):
    """Was `test_include_mechanics_refuses` (Step 3.2). Mechanics now exists,
    but it is meaningless without an equilibrated sigma_0: the defaults that
    would otherwise apply are sigma_0 = 0 (a stress-free domain) and
    rho_0 = rho_b_ref (not the wet profile the FE warm-up equilibrated
    against). Both are silently wrong, so this raises rather than defaults."""
    with pytest.raises(ValueError, match="no sigma_0 attached"):
        call(net, bundle, include_mechanics=True)


def test_include_mechanics_runs_once_sigma0_is_attached(net, bundle):
    from src.sigma0 import attach_sigma0
    coll = attach_sigma0(sample_interior(N, SEED))
    b = dict(bundle, coll=coll)
    a, parts_off = call(net, b)
    c, parts_on = call(net, b, include_mechanics=True)
    assert "pde_mech" in parts_on and "pde_mech" not in parts_off
    assert float(c.detach()) > float(a.detach())


def test_mechanics_off_is_bit_identical_to_before(net, bundle):
    """The ablation depends on the hydraulic half being untouched by the
    presence of the mechanical machinery."""
    from src.sigma0 import attach_sigma0
    b = dict(bundle, coll=attach_sigma0(sample_interior(N, SEED)))
    a, _ = call(net, bundle)
    c, _ = call(net, b, include_mechanics=False)
    assert float(c.detach()) == float(a.detach())


def test_no_mechanical_term_leaks_into_parts_when_off(net, bundle):
    """Renamed from `test_no_mechanical_term_leaks_into_parts`. The claim is
    now conditional: with include_mechanics=False nothing mechanical may
    appear, which is what makes the one-way run in the Step 6.2 ablation a
    genuine one-way run."""
    _, parts = call(net, bundle, per_term=True)
    for k in parts:
        assert "mech" not in k.lower(), f"unexpected mechanical part {k!r}"
        assert "disp" not in k.lower() or k.startswith("ic_"), (
            f"unexpected displacement part {k!r} outside the IC term"
        )


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
def test_gradients_reach_the_network(bundle):
    n = TinyNet(seed=1)
    total, _ = call(n, bundle)
    total.backward()
    grads = [p.grad for p in n.parameters()]
    assert all(g is not None for g in grads), "a parameter received no gradient"
    assert max(float(g.abs().max()) for g in grads) > 0.0, "all gradients zero"


def test_every_term_contributes_gradient(bundle):
    """Zeroing one term must change the gradient. Catches a term that is
    computed, reported, and then dropped from the sum."""
    for term in TERMS:
        n = TinyNet(seed=2)
        total, _ = total_loss(n, bundle["coll"], bundle["bcs"], bundle["ic"],
                              bundle["ifaces"], bundle["mats"],
                              weights={term: 0.0})
        total.backward()
        g = torch.cat([p.grad.reshape(-1) for p in n.parameters()])

        n2 = TinyNet(seed=2)
        full, _ = total_loss(n2, bundle["coll"], bundle["bcs"], bundle["ic"],
                             bundle["ifaces"], bundle["mats"])
        full.backward()
        g_full = torch.cat([p.grad.reshape(-1) for p in n2.parameters()])

        assert not torch.allclose(g, g_full), (
            f"dropping {term} did not change the gradient; it may not be in "
            f"the sum"
        )