"""
tests/test_sampling.py
======================

Guards the collocation sampler against the two failure modes that would be
invisible in a loss curve:

  1. Points outside the domain. The residual would be evaluated on geometry
     that does not exist, K(psi) would be called with a material tag from the
     wrong stratum, and training would converge to something plausible and
     wrong. `test_all_interior_points_are_inside` is the assertion.

  2. Weights that do not restore the area fractions. Stratifying 0.25/0.35/0.40
     against true fractions of 3.07/9.68/87.25% biases L_PDE by up to 17x per
     point. `test_weights_make_the_estimator_unbiased` checks the correction
     numerically rather than trusting the algebra.

Run:  pytest tests/test_sampling.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.nondim import SCALES
from src.sampling import (SAMPLE_SHARES, domain_bbox, load_initial,
                          sample_boundary, sample_interior)

try:
    from src.step21_geometry import geometry as g
except ModuleNotFoundError:                                   # pragma: no cover
    import geometry as g

N = 3000
SEED = 20260808
Z_WT = 197.0


def _physical(c):
    """Dimensionless tensors -> physical (x, z) numpy, for geometry calls."""
    x = c.x.detach().numpy().ravel() * SCALES.L_ref
    z = c.z.detach().numpy().ravel() * SCALES.L_ref
    return x, z


# ---------------------------------------------------------------------------
# Interior points
# ---------------------------------------------------------------------------
def test_all_interior_points_are_inside():
    c = sample_interior(N, seed=SEED)
    x, z = _physical(c)
    inside = np.asarray(g.inside_domain(x, z), dtype=bool)
    assert inside.all(), f"{(~inside).sum()} of {len(x)} points outside the domain"


def test_bbox_brackets_the_domain():
    """The scan must not clip the domain, or the corners never get sampled."""
    x0, x1, z0, z1 = domain_bbox()
    assert z0 < g.Z_BASE, f"bbox floor {z0} is above Z_BASE {g.Z_BASE}"
    c = sample_interior(N, seed=SEED)
    x, z = _physical(c)
    assert x.min() >= x0 and x.max() <= x1
    assert z.min() >= z0 and z.max() <= z1


def test_counts_follow_sample_shares():
    c = sample_interior(N, seed=SEED)
    counts = c.counts()
    for tag, share in SAMPLE_SHARES.items():
        assert abs(counts[tag] - N * share) <= 1, f"{tag}: {counts[tag]}"


def test_tags_agree_with_geometry():
    """The stored tag must match what material_tag says at that point —
    otherwise the residual gets the wrong SWCC for the wrong stratum."""
    c = sample_interior(N, seed=SEED)
    x, z = _physical(c)
    fresh = np.asarray(g.material_tag(x, z)).astype(str)
    assert (fresh == c.tag).all()


def test_weights_have_unit_mean():
    c = sample_interior(N, seed=SEED)
    assert abs(c.w.mean().item() - 1.0) < 1e-12
    assert (c.w > 0).all()


def test_weights_make_the_estimator_unbiased():
    """The point of the weights, checked numerically.

    Estimate the Mk_d area fraction two ways: from an unstratified sample
    (where the count IS the fraction) and from the weighted stratified
    sample. If the weights are right the two agree.
    """
    plain = sample_interior(8000, seed=SEED, stratify=False)
    strat = sample_interior(8000, seed=SEED, stratify=True)

    frac_plain = float((plain.tag == "Mk_d").mean())
    w = strat.w.detach().numpy().ravel()
    frac_weighted = float(w[strat.tag == "Mk_d"].sum() / w.sum())

    assert abs(frac_weighted - frac_plain) < 0.01, \
        f"weighted {frac_weighted:.4f} vs unstratified {frac_plain:.4f}"


def test_stratification_actually_oversamples_the_weak_zone():
    """Not vacuous: confirm Mk_d really is oversampled relative to its area.
    If this fails, stratify=True is doing nothing and the weights are all 1.

    Section 5 fractions are Mk 3.07 / Mk_d 9.68 / Tm 87.25 %, so the
    stratified 0.35 share is a 3.6x oversample of Mk_d. The 2.5x threshold
    leaves margin for Monte Carlo noise in the area estimate.
    """
    plain = sample_interior(4000, seed=SEED, stratify=False)
    strat = sample_interior(4000, seed=SEED, stratify=True)
    ratio = (strat.tag == "Mk_d").mean() / (plain.tag == "Mk_d").mean()
    assert ratio > 2.5, f"Mk_d oversampled only {ratio:.2f}x"


def test_time_within_window():
    c = sample_interior(N, seed=SEED, t_max=30.0)
    t = c.t.detach().numpy()
    assert t.min() >= 0.0 and t.max() <= 30.0


def test_inputs_are_differentiable_leaves():
    """residuals.grad() needs leaf tensors with requires_grad, or every
    derivative comes back as zeros and the PDE loss is silently constant."""
    c = sample_interior(500, seed=SEED)
    for name, t in (("x", c.x), ("z", c.z), ("t", c.t)):
        assert t.requires_grad, f"{name} is not differentiable"
        assert t.is_leaf, f"{name} is not a leaf"
        assert t.dtype == torch.float64
    assert not c.w.requires_grad


def test_deterministic_given_seed():
    a = sample_interior(1000, seed=1234)
    b = sample_interior(1000, seed=1234)
    assert torch.equal(a.x, b.x) and torch.equal(a.z, b.z)
    c = sample_interior(1000, seed=4321)
    assert not torch.equal(a.x, c.x)


# ---------------------------------------------------------------------------
# Boundary points
# ---------------------------------------------------------------------------
def test_all_six_segments_present():
    b = sample_boundary(600, seed=0)
    expected = {"natural_ground", "bench", "cut_face",
                "pit_floor", "base", "far_field_f1"}
    assert set(b) == expected, f"missing {expected - set(b)}"
    for seg, c in b.items():
        assert len(c) > 0, f"{seg} is empty"
        assert c.x.requires_grad


# ---------------------------------------------------------------------------
# Initial condition cache
# ---------------------------------------------------------------------------
def _cache_missing() -> bool:
    import pathlib
    return not pathlib.Path("src/step21_geometry/ic_cache.npz").exists()


@pytest.mark.skipif(_cache_missing(),
                    reason="ic_cache.npz absent; regenerate with 14_ic_checks.py")
def test_ic_cache_loads_and_is_hydrostatic():
    """psi0 in the cache must equal -(z - 197) exactly. Cross-checks
    sampling.load_initial against initial.psi_initial without importing it."""
    c, targets = load_initial()
    assert {"psi0", "sig_v", "sig_h"} <= set(targets)
    _, z = _physical(c)
    expected = -(z - Z_WT)
    assert np.abs(targets["psi0"] - expected).max() < 1e-6
    assert (c.t == 0).all(), "IC points must sit at t* = 0"
    assert (targets["psi0"] < 0).all(), "Section 5 is unsaturated everywhere"


if __name__ == "__main__":
    c = sample_interior(5000, seed=SEED)
    x, z = _physical(c)
    print(f"n = {len(c)}   inside = {np.asarray(g.inside_domain(x, z)).all()}")
    print(f"counts   {c.counts()}")
    w = c.w.detach().numpy().ravel()
    for tag in sorted(SAMPLE_SHARES):
        m = c.tag == tag
        print(f"  {tag:5} share {m.mean():.3f}  weight {w[m][0]:.4f}  "
              f"area {w[m].sum() / w.sum():.4f}")