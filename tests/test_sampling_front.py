"""Tests for src/sampling_front.py.

The load-bearing claim in that module's docstring is that its weights leave
`(w * f).mean()` an unbiased estimate of the same domain integral
`sample_interior` estimates, so the loss VALUE stays comparable across samplers
and switching sampler does not silently change the objective. That claim is
what `test_weights_restore_the_area_average` pins, with
`test_dropping_the_weights_changes_the_objective` as the negative control --
without it the first test would pass for a sampler whose weights did nothing.

Day 27 measurement behind the tolerances (10 seeds, N = 4000, against a
400k-point direct uniform-in-area reference): neither sampler is detectably
biased, but the front sampler's standard error on smooth integrands is 2.8x to
6.9x smaller than `sample_interior`'s. Importance sampling working as intended.
The failure that killed it for seed measurement (scripts/residual_tail.py) is
specific to the near-singular van Genuchten residual, not to the weights.
"""
import numpy as np
import pytest

from src.nondim import SCALES
from src.sampling import _rejection_sample, domain_bbox, sample_interior
from src.sampling_front import (N_BANDS, SUCTION_EDGES, SURFACE_BAND_M, Z_WT,
                                band_of, sample_interior_front)

try:
    import geometry as g
except ModuleNotFoundError:
    from src.step21_geometry import geometry as g

N = 3000
SEEDS = (11, 22)

# Smooth test integrands. `exp` decays over 40 m of suction, so it weights the
# conductive bottom of the domain the way the Richards residual's live band
# does, without the van Genuchten singularity that makes the real residual
# untestable this way.
INTEGRANDS = {
    "z":   lambda x, z: z,
    "z^2": lambda x, z: z ** 2,
    "exp": lambda x, z: np.exp(-(z - Z_WT) / 40.0),
}


def _phys(coll):
    x = coll.x.detach().reshape(-1).numpy() * SCALES.L_ref
    z = coll.z.detach().reshape(-1).numpy() * SCALES.L_ref
    w = coll.w.detach().reshape(-1).numpy()
    return x, z, w


def _weighted(coll, f):
    x, z, w = _phys(coll)
    return float((w * f(x, z)).mean() / w.mean())


@pytest.fixture(scope="module")
def truth():
    """Direct uniform-in-area reference, independent of either sampler."""
    rng = np.random.default_rng(0)
    x, z = _rejection_sample(200_000, domain_bbox(), rng)
    return {k: float(f(x, z).mean()) for k, f in INTEGRANDS.items()}


@pytest.mark.parametrize("name", list(INTEGRANDS))
def test_weights_restore_the_area_average(truth, name):
    """The stratified draw estimates the SAME integral, not a reweighted one.

    Tolerance is 2%, comfortably above the measured Monte-Carlo scatter at
    this N and far below the 8-45% error dropping the weights produces (see
    the negative control below).
    """
    f = INTEGRANDS[name]
    got = np.mean([_weighted(sample_interior_front(N, sd), f) for sd in SEEDS])
    assert got == pytest.approx(truth[name], rel=0.02), (
        f"{name}: front sampler gives {got:.6g} against a uniform-in-area "
        f"reference of {truth[name]:.6g}. The band weights no longer restore "
        "the area fractions, so L_PDE is integrating a different measure and "
        "its value is not comparable to a sample_interior run.")


@pytest.mark.parametrize("name", ["exp"])
def test_dropping_the_weights_changes_the_objective(truth, name):
    """Negative control for the test above, and a guard on the flag.

    `restore_weights=False` is a legitimate modelling choice -- the residual
    integrated against the sampling density rather than against area -- but it
    is a DIFFERENT objective and must not be reachable by accident. If this
    test ever starts failing, the flag has stopped doing anything and the two
    options have silently merged.
    """
    f = INTEGRANDS[name]
    got = np.mean([_weighted(sample_interior_front(N, sd, restore_weights=False), f)
                   for sd in SEEDS])
    rel = abs(got - truth[name]) / truth[name]
    assert rel > 0.10, (
        f"{name}: unweighted estimate {got:.6g} is within {100 * rel:.1f}% of "
        f"the area average {truth[name]:.6g}. restore_weights=False is "
        "supposed to be a different measure; if it is not, the weights are "
        "not load-bearing and the sampler is not doing what it claims.")


def test_the_surface_band_takes_precedence_over_elevation():
    """A point 1 m under the crest is in the infiltration layer, not band 5.

    This is the whole reason the surface band exists: the rain BC drives psi
    toward 0 at the ground surface whatever the elevation, and the ground
    surface spans z = 271..358 m, so no elevation band can contain it.
    """
    x = np.array([150.0, 150.0])
    z_top = g.z_ground(x[0])
    z = np.array([z_top - 1.0, z_top - 50.0])
    b = band_of(x, z)
    assert b[0] == 0, "a point 1 m below the ground surface must be band 0"
    assert b[1] != 0, "a point 50 m below the ground surface must not be"
    assert z_top - Z_WT > SUCTION_EDGES[-2], (
        "this test is only meaningful where the crest sits in the DEEPEST "
        "suction band; the geometry or SUCTION_EDGES has changed")


def test_every_band_is_populated_at_its_requested_share():
    coll = sample_interior_front(N, SEEDS[0])
    x, z, _ = _phys(coll)
    counts = np.bincount(band_of(x, z), minlength=N_BANDS)
    assert (counts > 0).all(), f"empty band(s): {counts}"
    # Default shares are uniform across bands; allow slack for points that
    # land on a band edge after the L_ref round trip.
    assert counts.min() / counts.max() > 0.9, (
        f"bands are not equally populated: {counts}")


def test_the_front_sampler_moves_the_budget_off_the_dead_zone():
    """Pins the Day 27 headline: uniform sampling starves the live band.

    `scripts/inert_fraction.py` at reach >= 0.1 m over 30 days gives 5.29%
    live for `sample_interior` and 37.93% for this sampler at N = 10000. Here
    the proxy is the same partition the sampler is built on -- the two
    shallowest suction bands plus the surface band, which is where every live
    point in that measurement fell.
    """
    def shallow_frac(coll):
        x, z, _ = _phys(coll)
        b = band_of(x, z)
        return float(np.isin(b, (0, 1, 2)).mean())

    uni = np.mean([shallow_frac(sample_interior(N, sd)) for sd in SEEDS])
    fro = np.mean([shallow_frac(sample_interior_front(N, sd)) for sd in SEEDS])
    assert uni < 0.20, f"uniform sampling now reaches {uni:.3f} of the live bands"
    assert fro > 3 * uni, (
        f"front sampler puts {fro:.3f} in the live bands against uniform's "
        f"{uni:.3f}; it is no longer concentrating anything")


def test_shares_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        sample_interior_front(500, 1, shares=np.full(N_BANDS, 0.5))
    with pytest.raises(ValueError, match="length"):
        sample_interior_front(500, 1, shares=np.array([0.5, 0.5]))


def test_the_water_table_is_below_the_domain():
    """Guards the Day 27 correction, not the sampler.

    Z_WT = 197 m and Z_BASE = 200 m, so Section 5 is unsaturated everywhere and
    'collocation near the water table' has no referent inside the domain. If
    this ever fails, the geometry or the IC has moved and every liveness
    argument in sampling_front.py and inert_fraction.py needs rechecking.
    """
    assert Z_WT < g.Z_BASE, (
        f"Z_WT = {Z_WT} is no longer below Z_BASE = {g.Z_BASE}; there is now a "
        "saturated region in the domain and the suction banding is wrong")
    assert SUCTION_EDGES[0] <= g.Z_BASE - Z_WT, (
        "the shallowest suction band starts above the domain floor, so the "
        "floor's own points fall outside the banding")
    assert SURFACE_BAND_M > 0.31, (
        "the surface band is now thinner than the 0.31 m saturated 30-day "
        "diffusion length in the marls it exists to bracket")