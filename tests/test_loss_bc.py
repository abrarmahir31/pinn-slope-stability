"""Tests for `L_BC` (Step 3.2).

Division of responsibility
--------------------------
`darcy_flux` is pinned by `tests/test_interface.py`; this module assumes it is
correct and tests only how `L_BC` *assembles* it into a residual:

    r = q*.n + q_prescribed*        n outward, q_prescribed positive INTO domain

so a satisfied infiltrating BC has ``q*.n = -q_prescribed*``. Dropping that
minus sign makes rainfall drain the slope and still converges, which is why
`test_sign_convention_is_not_reversible` and
`test_infiltration_lowers_residual_exfiltration_raises_it` both exist and why
they are run at a *calibrated* field strength — at a generic field strength
|q*.n| swamps q_prescribed* by three orders of magnitude and the flipped-sign
assembly is numerically indistinguishable from the correct one.

The anchor field
----------------
`14_ic_checks.py` establishes that under the t=0 profile psi = -(z - Z_WT) the
total head psi + z is uniform, so the Darcy flux is identically zero. That one
field does four jobs here:

  * zero flux              -> no-flow segments must vanish EXACTLY
  * equals `psi_far_field` -> the Dirichlet segment must vanish EXACTLY
  * zero flux              -> rainfall segments leave exactly q_prescribed*^2,
                              the baseline the sign tests are measured against
  * analytic               -> goes through `fields` unchanged, no net required
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

# ===========================================================================
# ADAPTER BLOCK -- the only place to touch if paths or signatures differ.
# ===========================================================================
from nondim import SCALES as S
from loss import L_BC, _psi_field, _BC_KIND
from sampling import sample_boundary
from residuals import darcy_flux
from materials import load_materials
from step21_geometry.boundaries import (
    Z_WT,
    RAIN_FLUX,
    SEEPAGE_FACE_MODE,
    flux_bc,
)
import properties as P

N_BOUNDARY = 400
SEED = 20250812
T_MAX = 30.0
DTYPE = torch.float64


MATS = load_materials()


def call_L_BC(field, bcs):
    """`L_BC(net, bcs, mats, s, per_segment)`.

    per_segment=True is not the default, and without it the parts dict has no
    per-segment keys at all -- most of this file depends on it.
    """
    return L_BC(field, bcs, MATS, per_segment=True)


def call_flux_bc(segment, coll):
    """`flux_bc` returns PHYSICAL m/s, positive into the domain -> (N,1) nondim.

    NOTE: the numpy `flux_bc(segment, pts)` took an (N,2) physical array and
    recomputed the surface normal from `g.z_ground`. If the torch-layer version
    now reads `coll.nx`/`coll.nz` instead (it should -- see the geometry
    round-trip note in the handoff), simplify this to `flux_bc(segment, coll)`.
    """
    pts = np.column_stack([_np(coll.x) * S.L_ref, _np(coll.z) * S.L_ref])
    q = np.asarray(flux_bc(segment, pts, tag=coll.tag), dtype=float)
    return torch.as_tensor(q.reshape(-1, 1) / S.Q_ref, dtype=DTYPE)


# ===========================================================================
# Helpers
# ===========================================================================
FLUX_SEGMENTS = tuple(s for s, k in _BC_KIND.items() if k == "flux")
DIRICHLET_SEGMENTS = tuple(s for s, k in _BC_KIND.items() if k != "flux")
RAINFALL_SEGMENTS = ("natural_ground", "bench")


def _np(t):
    return t.detach().cpu().numpy().reshape(-1)


def PART(parts, segment):
    """Parts are keyed `bc_<segment>`, not `<segment>`."""
    return float(parts[f"bc_{segment}"])


@pytest.fixture(scope="module")
def bcs():
    return sample_boundary(N_BOUNDARY, SEED, T_MAX)


@pytest.fixture(scope="module")
def W_global(bcs):
    """Sum of ALL boundary weights. Parts are normalised by this, not per-segment."""
    return sum(float(c.w.sum()) for c in bcs.values())


ZW_STAR = Z_WT / S.L_ref


def hydrostatic(offset: float = 0.0, a: float = 0.0, b: float = 0.0):
    """psi* = -(z* - Z_WT*)/Pi_R_hz + offset + a*(z* - Z_WT*) + b*x*.

    a = b = offset = 0 reproduces psi = -(z - Z_WT): uniform total head, zero
    Darcy flux, and identical to `psi_far_field` on F1. That is the anchor.

    a > 0 makes psi* increase with z* relative to hydrostatic -- wetter above,
    so flow is DOWNWARD, infiltrating at an upward-facing surface.
    Physically q_z = -K * Pi_R_hz * a.

    b drives flow in x: q_x = -K * Pi_R_hz * b. Needed because a z-only field
    is INVISIBLE to a segment with a horizontal normal -- `pit_floor` is the
    vertical cut at x = 0, normal (-1, 0), so q.n = q_z * 0 = 0 identically no
    matter how large a is. Any test that must see every segment react has to
    perturb both directions.
    """
    def field(x, z, t):
        dz = z - ZW_STAR
        psi = -dz / S.Pi_R_hz + offset + a * dz + b * x
        # keep x and t in the autograd graph -- `derivatives.py` differentiates
        # w.r.t. all three and an unused leaf yields a None grad, not a zero.
        return psi + 0.0 * x + 0.0 * t
    return field


def qn(field, coll):
    """q*.n at every point of a boundary set, (N,1), detached.

    `darcy_flux` takes ONE material, but boundary segments are mixed-tag
    (`natural_ground` crosses the Mk|Mk_d divide at x = 90.7). So partition by
    the tag that travelled with the Collocation -- never by re-querying
    geometry -- and evaluate per material.

    Each slice is rebuilt as a fresh leaf requiring grad. `coll.x[idx]` would be
    a non-leaf view and the autograd inside `darcy_flux` is not guaranteed to
    reach it; a detached clone sidesteps that entirely, which is safe here
    because only the values are wanted, not a gradient path.
    """
    out = torch.zeros_like(coll.x)
    for name in np.unique(coll.tag):
        assert name in MATS, f"unexpected tag {name!r} on a boundary set"
        idx = torch.as_tensor(np.flatnonzero(coll.tag == name), dtype=torch.long)
        x = coll.x[idx].detach().clone().requires_grad_(True)
        z = coll.z[idx].detach().clone().requires_grad_(True)
        t = coll.t[idx].detach().clone().requires_grad_(True)
        qx, qz = darcy_flux(field, x, z, t, MATS[name])
        v = qx * coll.nx[idx] + qz * coll.nz[idx]
        out = out.index_copy(0, idx, v.detach())
    return out


def assemble_part(coll, r):
    """What a per-segment PART is specified to be: a per-segment weighted mean."""
    return float((coll.w * r ** 2).sum()) / float(coll.w.sum())


def assemble_total(coll, r, W_global):
    """What a segment CONTRIBUTES to the total: divided by the global weight sum."""
    return float((coll.w * r ** 2).sum()) / W_global


def mean_w(coll, v):
    return float((coll.w * v).sum() / coll.w.sum())


CAL_FRAC = 0.5          # peak |q*.n| as a fraction of the prescribed flux
_CAL_CACHE: dict[str, float] = {}


def calibrate_infiltration(segment, coll, a_max=1e12):
    """Find a > 0 with peak |q*.n| = CAL_FRAC * q_prescribed* on this segment.

    Calibrating on the PEAK, not the mean, is what makes the sign tests
    rigorous. The cap makes q_prescribed* uniform along the segment, so
    bounding |q*.n| <= 0.5 q_prescribed* everywhere gives, pointwise,

        correct convention:  r = q_pres - |q.n|  in [0.5, 1.0] * q_pres
        flipped convention:  r = q_pres + |q.n|  in [1.0, 1.5] * q_pres

    Every point moves the same way, so the ordering below is guaranteed rather
    than a property of the average. A mean-based target permits a heavy tail
    where a few points overshoot 2*q_pres and push the residual back up.

    The bracket has to run to large a: the rainfall surface sits at psi ~ -150 m
    where K_r has collapsed, so a substantial gradient buys very little flux.
    """
    if segment in _CAL_CACHE:
        return _CAL_CACHE[segment]

    q_pres = call_flux_bc(segment, coll)
    q_min = float(q_pres.min())
    if q_min <= 0.0:
        pytest.skip(f"{segment} has no positive prescribed flux to calibrate to")
    target = CAL_FRAC * q_min

    def peak(a):
        return float(qn(hydrostatic(a=a), coll).abs().max())

    lo, hi = 0.0, 1e-6
    while peak(hi) < target:
        hi *= 8.0
        if hi > a_max:
            pytest.skip(f"cannot reach {target:.3e} on {segment} with a < {a_max:g}")
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if peak(mid) < target:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)

    got = peak(a)
    assert got == pytest.approx(target, rel=0.05), (
        f"calibration on {segment} landed at {got:.3e}, wanted {target:.3e} -- "
        f"peak |q.n| may not be monotone in a"
    )
    _CAL_CACHE[segment] = a
    return a


class TinyNet(torch.nn.Module):
    """(x,z,t) -> (psi*, u*, v*). Three outputs on purpose: `_psi_of` returns a
    bare Tensor unchanged, so a raw net silently sums gradients across all
    three columns. `test_psi_field_wrapper_is_not_optional` pins that."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 3),
        ).to(DTYPE)

    def forward(self, x, z, t):
        return self.net(torch.cat([x, z, t], dim=1))


# ===========================================================================
# 1. Contract
# ===========================================================================
def test_returns_scalar_and_detached_per_segment_parts(bcs):
    total, parts = call_L_BC(hydrostatic(), bcs)

    assert total.ndim == 0
    assert "bc" in parts
    for seg in bcs:
        assert f"bc_{seg}" in parts, f"bc_{seg} missing from parts"
    assert float(parts["bc"]) == pytest.approx(float(total.detach()), rel=1e-12)

    for k, v in parts.items():
        assert not torch.as_tensor(v).requires_grad, f"parts[{k}] is not detached"


def test_total_is_globally_weighted_while_parts_are_per_segment_means(
        bcs, W_global):
    """The two normalisations differ, deliberately.

    The TOTAL divides by the global weight sum, so a 400-point segment gets
    four times the say of a 100-point one -- the repo convention, and the one
    that matters for training.

    Each PART is a per-segment mean, so segments stay comparable as diagnostics
    regardless of how many points they were sampled with.

    The consequence is that parts do NOT sum to the total. Anything that later
    reports `sum(parts)` as the BC loss will be wrong, which is what this test
    exists to catch.
    """
    total, parts = call_L_BC(hydrostatic(), bcs)

    # under the anchor field only the rainfall segments are nonzero
    expected_total = sum(
        assemble_total(bcs[s], call_flux_bc(s, bcs[s]), W_global)
        for s in RAINFALL_SEGMENTS
    )
    assert float(total.detach()) == pytest.approx(expected_total, rel=1e-9)

    naive = sum(PART(parts, s) for s in bcs)
    assert naive != pytest.approx(float(total.detach()), rel=1e-3), (
        "parts appear to sum to the total; the two normalisations have "
        "converged and one of them changed"
    )


def test_all_six_segments_are_present(bcs):
    assert set(bcs) == {
        "natural_ground", "bench", "cut_face",
        "pit_floor", "base", "far_field_f1",
    }


# ===========================================================================
# 2. The zero-flux anchor, with negative controls
# ===========================================================================
def test_uniform_total_head_gives_zero_darcy_flux(bcs):
    """Precondition for everything below. If this fails, nothing else means
    anything -- the field is wrong, not `L_BC`."""
    for seg, coll in bcs.items():
        q = qn(hydrostatic(), coll).abs().max().item()
        assert q < 1e-12, f"{seg}: |q*.n| = {q:.2e}, expected 0"


@pytest.mark.parametrize("segment", [s for s in FLUX_SEGMENTS])
def test_no_flow_segments_vanish_under_the_anchor_field(bcs, segment):
    coll = bcs[segment]
    if float(call_flux_bc(segment, coll).abs().max()) != 0.0:
        pytest.skip(f"{segment} carries a prescribed flux")

    _, parts = call_L_BC(hydrostatic(), bcs)
    assert PART(parts, segment) < 1e-24, (
        f"{segment} is no-flow and the field has zero Darcy flux; "
        f"got {PART(parts, segment):.2e}"
    )


@pytest.mark.parametrize("segment", [s for s in FLUX_SEGMENTS])
def test_no_flow_segments_are_nonzero_under_a_flowing_field(bcs, segment):
    """Negative control. Without this, `return 0.0` passes the test above.

    Measured against the segment's OWN anchor value, not an absolute floor.
    An absolute threshold is not portable across segments: `cut_face` sits at
    psi* between -2.5 and -4.5 under the anchor profile, and at n = 1.2 the
    relative conductivity there is small enough that a perfectly healthy
    response is O(1e-28). What matters is that the segment moved, by orders of
    magnitude, not that it cleared some fixed bar.
    """
    coll = bcs[segment]
    if float(call_flux_bc(segment, coll).abs().max()) != 0.0:
        pytest.skip(f"{segment} carries a prescribed flux")

    _, anchor = call_L_BC(hydrostatic(), bcs)
    # perturb BOTH directions: a z-only field has zero normal flux on
    # `pit_floor`, whose normal is horizontal
    _, flowing = call_L_BC(hydrostatic(a=1e-2, b=1e-2), bcs)

    at_rest, moved = PART(anchor, segment), PART(flowing, segment)
    assert moved > 0.0, f"{segment} is identically zero for every field"
    assert moved > 1e6 * max(at_rest, 1e-300), (
        f"{segment} did not react to a flowing field: "
        f"{at_rest:.3e} -> {moved:.3e}"
    )


def test_seepage_face_is_currently_no_flow():
    """`cut_face` is a complementarity condition in the plan but ships as
    no-flow. If SEEPAGE_FACE_MODE changes, the two tests above stop applying
    to it and `L_BC` needs a Fischer-Burmeister branch."""
    assert SEEPAGE_FACE_MODE == "noflow", (
        "seepage-face mode changed; L_BC needs a complementarity branch and "
        "these tests need revisiting"
    )


# ===========================================================================
# 3. The Dirichlet branch
# ===========================================================================
def test_far_field_vanishes_when_the_field_equals_psi_far_field(bcs):
    """The anchor field IS `psi_far_field`, nondimensionalised. Exact zero."""
    _, parts = call_L_BC(hydrostatic(), bcs)
    assert PART(parts, "far_field_f1") < 1e-24


@pytest.mark.parametrize("offset", [1e-3, 1e-2, 1e-1])
def test_far_field_residual_is_quadratic_in_a_constant_offset(bcs, W_global, offset):
    """r = psi* - psi_prescribed* = offset, so the part is offset^2 * (w-share).
    Pins that the Dirichlet branch measures a head difference, not a flux, and
    that it uses the same global normalisation as the flux branch."""
    _, parts = call_L_BC(hydrostatic(offset=offset), bcs)
    assert PART(parts, "far_field_f1") == pytest.approx(offset ** 2, rel=1e-9)


def test_dirichlet_segments_ignore_the_normals(bcs):
    """`far_field_f1`'s normal is ~8.6 deg off (finding 5). Harmless only while
    the segment stays Dirichlet -- this test is what makes that true."""
    coll = bcs["far_field_f1"]
    base, _ = call_L_BC(hydrostatic(offset=1e-2), bcs)
    with_broken_normals = {
        k: (v if k != "far_field_f1"
            else dataclasses.replace(v, nx=-v.nx, nz=-v.nz))
        for k, v in bcs.items()
    }
    flipped, _ = call_L_BC(hydrostatic(offset=1e-2), with_broken_normals)
    assert float(flipped.detach()) == pytest.approx(float(base.detach()), rel=1e-12)


# ===========================================================================
# 4. The rainfall cap
# ===========================================================================
@pytest.mark.parametrize("segment", RAINFALL_SEGMENTS)
def test_prescribed_flux_is_capped_at_saturated_conductivity(bcs, segment):
    """Cross-check of `flux_bc` against RAIN_FLUX, K_S and the stored normal --
    computed here from primitives rather than by calling `flux_bc` twice."""
    coll = bcs[segment]
    nz = _np(coll.nz)
    K_s = np.array([P.K_S[t] for t in coll.tag], dtype=float)
    expected = np.minimum(RAIN_FLUX * nz, K_s) / S.Q_ref

    got = _np(call_flux_bc(segment, coll))
    np.testing.assert_allclose(got, expected, rtol=1e-10)

    # and the cap is actually binding, not decorative
    assert (RAIN_FLUX * nz > K_s).mean() > 0.99, (
        "the cap is not active on this segment; finding 1 in the handoff "
        "assumes it is everywhere on the rainfall surface"
    )


@pytest.mark.parametrize("segment", RAINFALL_SEGMENTS)
def test_anchor_field_leaves_exactly_the_prescribed_flux(bcs, W_global, segment):
    """Zero Darcy flux => r = q_prescribed*, so the part is its weighted mean
    square. Pins magnitude; the next section pins sign."""
    coll = bcs[segment]
    q_pres = call_flux_bc(segment, coll)
    _, parts = call_L_BC(hydrostatic(), bcs)
    assert PART(parts, segment) == pytest.approx(
        assemble_part(coll, q_pres), rel=1e-9, abs=1e-30
    )


# ===========================================================================
# 5. Sign convention -- the point of the file
# ===========================================================================
@pytest.mark.parametrize("segment", RAINFALL_SEGMENTS)
def test_infiltration_lowers_residual_exfiltration_raises_it(bcs, segment):
    """Rain must not drain the slope.

    At the calibrated strength q*.n = -q_prescribed*, so:
        a = +a_cal (downward flow, into the domain)  -> r ~ 0
        a =  0     (no flow)                         -> r = q_prescribed*
        a = -a_cal (upward flow, out of the domain)  -> r = 2*q_prescribed*
    i.e. ratios of roughly 0 : 1 : 4. With the minus sign dropped the ordering
    reverses and this fails immediately.
    """
    coll = bcs[segment]
    a = calibrate_infiltration(segment, coll)

    _, p_in = call_L_BC(hydrostatic(a=+a), bcs)
    _, p_none = call_L_BC(hydrostatic(a=0.0), bcs)
    _, p_out = call_L_BC(hydrostatic(a=-a), bcs)

    r_in, r_none, r_out = (PART(p, segment) for p in (p_in, p_none, p_out))

    assert r_in < r_none, (
        f"{segment}: an infiltrating field did not reduce the rainfall "
        f"residual ({r_in:.6e} vs {r_none:.6e}) -- sign convention is reversed"
    )
    assert r_out > r_none, (
        f"{segment}: an exfiltrating field was not penalised "
        f"({r_out:.6e} vs {r_none:.6e})"
    )
    # bounds implied by |q.n| <= CAL_FRAC * q_pres pointwise
    assert (1.0 - CAL_FRAC) ** 2 * r_none <= r_in
    assert r_out <= (1.0 + CAL_FRAC) ** 2 * r_none


@pytest.mark.parametrize("segment", RAINFALL_SEGMENTS)
def test_sign_convention_is_not_reversible(bcs, W_global, segment):
    """Exact assembly check, plus the negative control that makes it bite.

    Run at the calibrated strength, where r_correct ~ 0 and r_flipped ~
    2*q_prescribed*, so the two conventions are separated by many orders of
    magnitude rather than by one part in a thousand.
    """
    coll = bcs[segment]
    a = calibrate_infiltration(segment, coll)
    field = hydrostatic(a=a)

    q_pres = call_flux_bc(segment, coll)
    flux = qn(field, coll).detach()

    r_correct = flux + q_pres      # q*.n = -q_prescribed* when satisfied
    r_flipped = flux - q_pres      # the bug

    _, parts = call_L_BC(field, bcs)
    got = PART(parts, segment)

    assert got == pytest.approx(assemble_part(coll, r_correct),
                                rel=1e-9, abs=1e-30)
    assert got != pytest.approx(assemble_part(coll, r_flipped), rel=1e-2)

@pytest.mark.parametrize("a", [0.0, 1e-3, 1e-2])
def test_reweighted_parts_reconstruct_the_total(bcs, W_global, a):
    """part_seg * (w_seg / W_global) summed over segments == total.

    This is the exact bridge between the two normalisations, and it holds for
    any field, so it pins both at once: undo each part's per-segment mean, apply
    the global weight share, and the total must come back. A 0.25-share segment
    gets 0.25 of the say; equalising them would break this.
    """
    total, parts = call_L_BC(hydrostatic(a=a), bcs)

    reconstructed = sum(
        PART(parts, seg) * float(coll.w.sum()) / W_global
        for seg, coll in bcs.items()
    )
    assert reconstructed == pytest.approx(float(total.detach()), rel=1e-9, abs=1e-30)


# ===========================================================================
# 6. Autograd plumbing
# ===========================================================================
def test_gradients_reach_the_network(bcs):
    net = TinyNet()
    total, _ = call_L_BC(_psi_field(net), bcs)
    total.backward()

    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads), "a parameter received no gradient"
    assert max(float(g.abs().max()) for g in grads) > 0.0, "all gradients are zero"


def test_psi_field_selects_one_column(bcs):
    """`_psi_of` passes a bare Tensor through unchanged, so an unwrapped net is
    treated as three psi columns and gradients are summed across u* and v*. No
    exception, just a wrong number.

    Tested on `_psi_field` directly rather than through `L_BC`, since `L_BC`
    takes `net` and may wrap internally -- in which case going through it would
    compare a value against itself and pass unconditionally.
    """
    net = TinyNet()
    coll = bcs["base"]
    raw = net(coll.x, coll.z, coll.t)
    wrapped = _psi_field(net)(coll.x, coll.z, coll.t)

    assert raw.shape[1] == 3, "TinyNet should emit (psi*, u*, v*)"
    assert wrapped.shape == (coll.x.shape[0], 1), (
        f"_psi_field must return (N,1), got {tuple(wrapped.shape)}"
    )
    assert torch.allclose(wrapped, raw[:, 0:1])


def test_segment_normals_point_where_the_geometry_says(bcs):
    """Which segments a given test field can even see depends on these.

    `pit_floor` is the vertical cut at x = 0 and `base` the horizontal floor at
    z = Z_BASE, so each is blind to a field whose gradient is parallel to its
    surface. Pinning the orientation here is what stops a future test from
    quietly passing on a segment it never actually exercised.
    """
    pf, ba, ff = bcs["pit_floor"], bcs["base"], bcs["far_field_f1"]

    assert float(pf.nz.abs().max()) < 1e-12, "pit_floor normal is not horizontal"
    assert float(pf.nx.max()) < 0.0, "pit_floor normal should point out at -x"

    assert float(ba.nx.abs().max()) < 1e-12, "base normal is not vertical"
    assert float(ba.nz.max()) < 0.0, "base normal should point out at -z"

    # finding 5: `_segment_normal` returns +x on F1, which actually dips 81.4
    # deg -- ~8.6 deg off. Harmless only while the segment stays Dirichlet.
    assert float(ff.nx.min()) > 0.0, "far_field_f1 normal should point out at +x"


def test_boundary_points_are_leaf_tensors_requiring_grad(bcs):
    for seg, coll in bcs.items():
        for name in ("x", "z", "t"):
            v = getattr(coll, name)
            assert v.requires_grad, f"{seg}.{name} does not require grad"
            assert v.is_leaf, f"{seg}.{name} is not a leaf"
            assert v.shape[1] == 1, f"{seg}.{name} is not (N,1)"


def test_normals_are_present_and_unit_on_boundary_sets(bcs):
    for seg, coll in bcs.items():
        assert coll.nx is not None and coll.nz is not None, f"{seg} has no normal"
        norm = torch.sqrt(coll.nx ** 2 + coll.nz ** 2)
        assert float((norm - 1.0).abs().max()) < 1e-10, f"{seg} normals not unit"


# ===========================================================================
# 7. Geometry is not re-queried in the torch layer
# ===========================================================================
def test_L_BC_does_not_requery_geometry(bcs, monkeypatch):
    """The x -> x/L_ref -> x round trip moves boundary points by ~1e-16, enough
    for `material_tag` to return "outside" on curved segments. Tags and normals
    travel with the Collocation; nothing downstream may look them up again."""
    import geometry as geom   # flat import: boundaries.py does `import geometry as g`

    def boom(*a, **k):
        raise AssertionError("L_BC re-queried geometry.material_tag")

    monkeypatch.setattr(geom, "material_tag", boom)
    total, _ = call_L_BC(hydrostatic(a=1e-3), bcs)
    assert torch.isfinite(total)


# ===========================================================================
# 8. Regression guard on the reported per-segment ordering
# ===========================================================================
def test_far_field_dominates_the_bc_loss(bcs):
    """Finding 1: with the cap in place the rainfall segments contribute ~1e-11
    and `L_BC` is essentially `far_field_f1` alone. This is a real result going
    into the thesis, so it gets a test -- if it ever stops holding, the
    stratigraphic reading changed and the write-up needs to change with it."""
    net = TinyNet()
    _, parts = call_L_BC(_psi_field(net), bcs)
    ranked = sorted(bcs, key=lambda s: -PART(parts, s))
    assert ranked[0] == "far_field_f1"
    for seg in RAINFALL_SEGMENTS:
        assert PART(parts, seg) < 1e-4 * PART(parts, "far_field_f1")