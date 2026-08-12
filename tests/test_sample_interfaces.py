"""Tests for `sample_interfaces` (Step 3.2, item 2).

The sampler's job is to put points exactly on `z_tm_top(x)`, tag which material
is on each side, and carry a correctly-oriented normal -- all of it computed
ONCE here, because the torch layer cannot re-query geometry without hitting the
x -> x/L_ref -> x round-trip problem.

Every geometric claim gets a negative control. A normal that is always (0,1)
satisfies "is unit" and "points up"; only the perpendicularity test against the
actual trace slope, plus the assertion that the contact is not flat, rules it
out.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nondim import SCALES as S
from sampling import (
    sample_interfaces,
    CONTACTS,
    INTERFACE_EPS_M,
    _tm_top_slope,
)
import geometry as g

N = 500
SEED = 20250812


@pytest.fixture(scope="module")
def ifaces():
    return sample_interfaces(N, SEED)


def _np(t):
    return t.detach().cpu().numpy().reshape(-1)


def phys(coll):
    """Physical (x, z) in metres."""
    return _np(coll.x) * S.L_ref, _np(coll.z) * S.L_ref


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_keyed_by_contact_like_sample_boundary(ifaces):
    assert set(ifaces) == {"mk_tm", "mkd_tm"}


def test_both_contacts_are_populated(ifaces):
    for key, coll in ifaces.items():
        assert coll.x.shape[0] > 0.9 * N, f"{key} lost more than 10% of points"


def test_points_are_leaf_tensors_requiring_grad(ifaces):
    for key, coll in ifaces.items():
        for name in ("x", "z", "t"):
            v = getattr(coll, name)
            assert v.requires_grad, f"{key}.{name} does not require grad"
            assert v.is_leaf, f"{key}.{name} is not a leaf"
            assert v.shape[1] == 1, f"{key}.{name} is not (N,1)"


def test_weights_and_normals_have_matching_shapes(ifaces):
    for key, coll in ifaces.items():
        n = coll.x.shape[0]
        for name in ("w", "nx", "nz"):
            assert getattr(coll, name).shape == (n, 1), f"{key}.{name} is not (N,1)"
        assert coll.tag.shape == (n,), f"{key}.tag is not (N,)"


def test_sampling_is_reproducible_and_seed_dependent():
    a = sample_interfaces(N, SEED)["mk_tm"]
    b = sample_interfaces(N, SEED)["mk_tm"]
    c = sample_interfaces(N, SEED + 1)["mk_tm"]

    assert torch.equal(a.x, b.x), "same seed gave different points"
    assert not torch.equal(a.x, c.x), "different seeds gave identical points"


# ---------------------------------------------------------------------------
# The points are on the contact
# ---------------------------------------------------------------------------
def test_points_lie_on_the_tm_top_trace(ifaces):
    for key, coll in ifaces.items():
        x, z = phys(coll)
        assert np.abs(z - g.z_tm_top(x)).max() < 1e-9, (
            f"{key}: points are off the trace"
        )


def test_points_are_inside_the_domain(ifaces):
    for key, coll in ifaces.items():
        x, z = phys(coll)
        assert (z > g.Z_BASE).all(), f"{key}: points at or below Z_BASE"
        assert (z < g.z_ground(x)).all(), f"{key}: points above the ground surface"
        assert (x <= g.x_f1(z)).all(), f"{key}: points beyond F1"


def test_contacts_split_at_the_mk_divide(ifaces):
    xm, _ = phys(ifaces["mk_tm"])
    xd, _ = phys(ifaces["mkd_tm"])
    assert xm.max() <= g.X_MK_DIVIDE, "mk_tm crosses the divide"
    assert xd.min() >= g.X_MK_DIVIDE, "mkd_tm crosses the divide"


def test_arclength_sampling_beats_uniform_in_x(ifaces):
    """Negative control on the sampler itself: if it were uniform in x, the
    spacing along the trace would be denser on the flat limbs than the steep
    ones. Compare the spread of arc-length gaps against x-gaps."""
    x, z = phys(ifaces["mkd_tm"])
    order = np.argsort(x)
    x, z = x[order], z[order]

    ds = np.hypot(np.diff(x), np.diff(z))
    dx = np.diff(x)
    assert ds.std() / ds.mean() < dx.std() / dx.mean(), (
        "arc-length spacing is no more uniform than x spacing; the sampler may "
        "be sampling uniformly in x"
    )


# ---------------------------------------------------------------------------
# Materials each side
# ---------------------------------------------------------------------------
def test_tag_holds_the_upper_material(ifaces):
    for key, coll in ifaces.items():
        assert set(np.unique(coll.tag)) == {CONTACTS[key]}, (
            f"{key}.tag should be the material ABOVE the contact"
        )


def test_material_above_and_below_are_as_the_key_says(ifaces):
    """Tm is the lower side by construction, which is why the Collocation only
    needs to carry the upper tag."""
    eps = INTERFACE_EPS_M
    for key, coll in ifaces.items():
        x, z = phys(coll)
        assert (g.material_tag(x, z + eps) == CONTACTS[key]).all()
        assert (g.material_tag(x, z - eps) == "Tm").all()


def test_no_points_land_outside_the_domain_after_offsetting(ifaces):
    """The eps offset must not push a point out of the domain -- that is what
    would produce an "outside" tag and a missing material in the torch layer."""
    eps = INTERFACE_EPS_M
    for key, coll in ifaces.items():
        x, z = phys(coll)
        for zz in (z + eps, z - eps):
            assert (g.material_tag(x, zz) != "outside").all(), f"{key}: offset left the domain"


def test_mk_mkd_contact_is_not_sampled(ifaces):
    """Mk|Mk_d carries no flux discontinuity while the marls share K_s, alpha
    and n. If that changes, this sampler needs a third contact and this test is
    the reminder."""
    assert "mk_mkd" not in ifaces
    assert len(ifaces) == 2


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------
def test_normals_are_unit(ifaces):
    for key, coll in ifaces.items():
        norm = torch.sqrt(coll.nx ** 2 + coll.nz ** 2)
        assert float((norm - 1.0).abs().max()) < 1e-12, f"{key} normals not unit"


def test_normals_point_up_out_of_tm(ifaces):
    """Sign convention: the normal points from the lower material (Tm) into the
    upper. `interface_flux_jump` uses the same normal on both sides, so
    flipping this flips the sign of the jump."""
    for key, coll in ifaces.items():
        assert float(coll.nz.min()) > 0.0, f"{key} has a downward normal"


def test_normals_are_perpendicular_to_the_trace(ifaces):
    """The test that a constant (0,1) normal fails."""
    for key, coll in ifaces.items():
        x, _ = phys(coll)
        dz = _tm_top_slope(x)
        tangent = np.column_stack([np.ones_like(dz), dz])
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

        dot = _np(coll.nx) * tangent[:, 0] + _np(coll.nz) * tangent[:, 1]
        assert np.abs(dot).max() < 1e-9, f"{key}: normal is not perpendicular"


def test_the_contact_is_not_flat(ifaces):
    """Negative control for the test above: if z_tm_top were horizontal, every
    normal would be (0,1) and perpendicularity would hold trivially."""
    tilted = 0
    for coll in ifaces.values():
        x, _ = phys(coll)
        tilted += int((np.abs(_tm_top_slope(x)) > 0.01).sum())
    assert tilted > 0.1 * N, (
        "the sampled contact is essentially horizontal; the perpendicularity "
        "test is vacuous"
    )


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
def test_t_spans_the_simulation_window(ifaces):
    for key, coll in ifaces.items():
        t = _np(coll.t)
        assert t.min() >= 0.0, f"{key} has negative t*"
        assert t.max() > 0.0, f"{key} has no transient points"


def test_t_max_is_respected():
    for key, coll in sample_interfaces(N, SEED, t_max=7.0).items():
        assert _np(coll.t).max() <= 7.0, f"{key} exceeded t_max"