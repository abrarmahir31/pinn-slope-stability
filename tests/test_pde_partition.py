"""L_PDE accumulates squared residuals globally, not as a mean of per-tag means.

Guards the docstring's contract: a 0.25-share material must not get equal
say with a 0.40-share one. The two formulations coincide iff all tags carry
equal weight, so this test is only meaningful on an unequal partition.
"""
import pytest, torch

from src.config import BOUNDS, tiny
from src.loss import L_PDE
from src.materials import load_materials
from src.model import PINN
from src.sampling import sample_interior


@pytest.fixture
def setup():
    torch.manual_seed(0)
    coll = sample_interior(400)
    mats = load_materials()
    net = PINN(tiny(), BOUNDS)
    return net, coll, mats


def _shares(coll):
    wsum = coll.w.sum()
    return {t: (coll.w[coll.tag == t].sum() / wsum).item()
            for t in sorted(set(coll.tag.tolist()))}


def test_partition_is_unequal(setup):
    """Precondition: without this, the test below proves nothing."""
    sh = list(_shares(setup[1]).values())
    assert len(sh) >= 2, "need >=2 tags in the collocation set"
    assert max(sh) - min(sh) > 0.05, f"shares too even to discriminate: {sh}"


def test_pde_equals_weight_share_combination(setup):
    net, coll, mats = setup
    pde, parts = L_PDE(net, coll, mats, per_tag=True)

    wsum = coll.w.sum()
    recombined = sum(parts[f"pde_richards_{t}"] * coll.w[coll.tag == t].sum()
                     for t in sorted(set(coll.tag.tolist()))) / wsum

    torch.testing.assert_close(recombined, pde.detach(), rtol=1e-12, atol=0.0)


def test_pde_is_not_a_mean_of_means(setup):
    net, coll, mats = setup
    pde, parts = L_PDE(net, coll, mats, per_tag=True)

    per_tag = [parts[f"pde_richards_{t}"] for t in sorted(set(coll.tag.tolist()))]
    mean_of_means = sum(per_tag) / len(per_tag)

    assert not torch.isclose(mean_of_means, pde.detach(), rtol=1e-6), (
        "L_PDE now matches a mean-of-means; the global weight-sum "
        "normalisation has been lost")


def test_summary_key_matches_scalar(setup):
    net, coll, mats = setup
    pde, parts = L_PDE(net, coll, mats, per_tag=True)
    torch.testing.assert_close(parts["pde_richards"], pde.detach(),
                               rtol=0.0, atol=0.0)