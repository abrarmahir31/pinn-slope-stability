"""One-at-a-time sensitivity, ablation and reproducibility — Step 6.2 support.

Scope note, so this file is not mistaken for more than it is: everything here
either (a) derives the perturbed PARAMETER SETS that each OAT arm needs, or
(b) aggregates FOS values that arms have already produced. It does not produce
a factor of safety and cannot. FOS comes from `scripts/ssr_sweep.py`, which is
blocked on the yield term not being wired into `loss.py` -- see
`docs/STEP5_STATUS.md`. Days 49-52 are entirely downstream of a number that
does not exist yet.

What IS useful now: the GSI arm of the OAT sweep re-derives m_b, s, a AND E_rm
each time, and getting that chain right is fiddly, order-dependent and easy to
half-do. It is also testable without a GPU, which is why it is here and tested
rather than inline in a script nobody ran.

Derivation chain for a GSI perturbation (all four must move together):

    GSI -> m_b = m_i * exp((GSI - 100) / (28 - 14D))       Hoek et al. 2002
        -> s   = exp((GSI - 100) / (9 - 3D))
        -> a   = 0.5 + (1/6)(exp(-GSI/15) - exp(-20/3))
        -> E_rm = E_i * (0.02 + (1 - D/2) / (1 + exp((60 + 15D - GSI)/11)))
                                                           Hoek-Diederichs 2006

Changing GSI and leaving E_rm at its baseline is the common half-measure. It is
wrong in a specific and misleading direction: GSI 40 vs 60 moves E_rm for marl
by roughly a factor of three, the softer mass strains more, and the
Kozeny-Carman feedback keys off volumetric strain. So an E_rm held fixed
suppresses exactly the coupling that Day 51 exists to measure, and the two
experiments end up contradicting each other for a reason that is not physics.
`test_gsi_arm_moves_modulus_and_strength_together` is the guard.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace

import numpy as np

from src import strength as st


# ---------------------------------------------------------------------------
# 1. Parameter derivation
# ---------------------------------------------------------------------------
def E_rm_hoek_diederichs(E_i: float, gsi: float, D: float) -> float:
    """Rock-mass deformation modulus, Hoek & Diederichs (2006) simplified form.

        E_rm = E_i * (0.02 + (1 - D/2) / (1 + exp((60 + 15D - GSI) / 11)))

    Reproduces every stored modulus in the Phase-1 dataset to four significant
    figures, and the Tm addendum in `properties.py` (E_i = 35.0 GPa, GSI 60,
    D 1 -> 4263.9 MPa). Pinned by `test_E_rm_reproduces_the_stored_moduli`.
    """
    return E_i * (0.02 + (1.0 - D / 2.0)
                  / (1.0 + math.exp((60.0 + 15.0 * D - gsi) / 11.0)))


@dataclass(frozen=True)
class StratumParams:
    """The full derived set for one stratum at one GSI."""
    tag: str
    sigma_ci: float
    gsi: float
    m_i: float
    D: float
    MR: float
    E_i: float
    m_b: float
    s: float
    a: float
    E_rm: float

    def ghb(self) -> st.GHBParams:
        return st.GHBParams(sigma_ci=self.sigma_ci, m_b=self.m_b,
                            s=self.s, a=self.a)


def derive_stratum(tag: str, sigma_ci: float, gsi: float, m_i: float,
                   D: float, MR: float) -> StratumParams:
    """Full GHB + modulus derivation from GSI. The OAT GSI arm calls this.

    `E_i = MR * sigma_ci` is the Deere modulus-ratio route used for every
    stratum here, including the Tm addendum, so the treatment stays uniform.
    Note E_i does NOT depend on GSI -- only E_rm does. Perturbing GSI must not
    move E_i, and `test_intact_modulus_is_invariant_to_gsi` says so.
    """
    g = st.ghb_from_gsi(sigma_ci=sigma_ci, gsi=gsi, m_i=m_i, D=D)
    E_i = MR * sigma_ci
    return StratumParams(tag=tag, sigma_ci=sigma_ci, gsi=gsi, m_i=m_i, D=D,
                         MR=MR, E_i=E_i, m_b=g.m_b, s=g.s, a=g.a,
                         E_rm=E_rm_hoek_diederichs(E_i, gsi, D))


# Section 5 baselines, from the Phase-1 dataset. Kept here so the OAT arms have
# something to perturb FROM; properties.py remains the source of truth for
# anything the solver reads.
BASELINE = {
    "marl_sekkoy":   dict(sigma_ci=17.9e6, gsi=50, m_i=4,  D=1.0, MR=175),
    "marl_weakzone": dict(sigma_ci=4.29e6, gsi=45, m_i=3,  D=1.0, MR=175),
    "coal":          dict(sigma_ci=7.80e6, gsi=40, m_i=18, D=1.0, MR=300),
    "tuffite":       dict(sigma_ci=3.59e6, gsi=26, m_i=15, D=1.0, MR=350),
    "Tm":            dict(sigma_ci=70.0e6, gsi=60, m_i=12, D=1.0, MR=500),
}


def baseline_stratum(tag: str) -> StratumParams:
    return derive_stratum(tag, **BASELINE[tag])


# ---------------------------------------------------------------------------
# 2. OAT experiment specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Arm:
    """One OAT arm: a named perturbation and the overrides it implies."""
    factor: str            # which input was moved
    level: str             # "low" / "base" / "high", for the tornado ordering
    label: str             # human-readable, goes on the plot
    overrides: dict        # what the solver must be launched with

    @property
    def is_base(self) -> bool:
        return self.level == "base"


def oat_arms(*, ks_frac: float = 0.20,
             rainfall_mm_hr=(10.0, 20.0, 40.0),
             gsi_levels=(40, 50, 60),
             gsi_stratum: str = "marl_sekkoy") -> list[Arm]:
    """The Step 6.2 arm list.

    Three factors, as the plan has them: coal-seam K_s at +/- `ks_frac`,
    rainfall intensity, and marl GSI. Returns the BASE arm once, first, so the
    tornado plot has a centre and the caller does not solve it three times.

    A caution about the rainfall factor specifically. `open_items.md` Finding 1
    records that the rainfall BC admits 0.02% of incident rain -- the whole
    rainfall surface sits at K_s = 1e-9 m/s and the flux cap binds on >99% of
    points. If that finding still holds, all three rainfall arms deliver a
    numerically identical boundary condition and the factor will come out of
    the tornado plot as a flat bar. That is a real and reportable result, not
    a bug, but decide in advance that you will report it -- a flat bar
    discovered on day 50 reads as a failed experiment, and the same bar
    predicted on day 49 reads as confirmation of Finding 1.
    """
    base_ks, base_rain = 1.0, 20.0
    arms = [Arm("baseline", "base", "baseline", {})]

    for lvl, f in (("low", 1.0 - ks_frac), ("high", 1.0 + ks_frac)):
        arms.append(Arm("coal_K_s", lvl, f"coal K_s x{f:.2f}",
                        {"ks_multiplier": {"coal": f}}))

    for r in rainfall_mm_hr:
        if r == base_rain:
            continue
        lvl = "low" if r < base_rain else "high"
        arms.append(Arm("rainfall", lvl, f"rain {r:.0f} mm/hr",
                        {"rain_mm_hr": r}))

    base_gsi = BASELINE[gsi_stratum]["gsi"]
    for g in gsi_levels:
        if g == base_gsi:
            continue
        lvl = "low" if g < base_gsi else "high"
        p = derive_stratum(gsi_stratum, **{**BASELINE[gsi_stratum], "gsi": g})
        arms.append(Arm("marl_GSI", lvl, f"{gsi_stratum} GSI {g}",
                        {"stratum_overrides": {gsi_stratum: {
                            "m_b": p.m_b, "s": p.s, "a": p.a, "E": p.E_rm}}}))

    return arms


def coupling_arms() -> list[Arm]:
    """Day 51. Three arms, per DECISIONS.md -- not two.

    The plan says "one-way vs two-way", but DECISIONS.md specifies
    `KCConfig(enabled=True)` (KC everywhere), `KC_DEFAULT` (marls only) and
    `KC_OFF` (one-way), and asks for per-stratum reporting. The middle arm is
    the configuration of record, so a two-arm comparison would contrast the
    baseline against neither of the alternatives it was chosen over.
    """
    return [
        Arm("coupling", "base", "KC marls only (KC_DEFAULT)", {"kc": "default"}),
        Arm("coupling", "high", "KC everywhere", {"kc": "all"}),
        Arm("coupling", "low", "KC off (one-way)", {"kc": "off"}),
    ]


# ---------------------------------------------------------------------------
# 3. Tornado assembly
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TornadoBar:
    factor: str
    low: float
    high: float
    base: float

    @property
    def span(self) -> float:
        return abs(self.high - self.low)

    @property
    def low_pct(self) -> float:
        return 100.0 * (self.low - self.base) / self.base

    @property
    def high_pct(self) -> float:
        return 100.0 * (self.high - self.base) / self.base


def tornado(results: dict[tuple[str, str], float]) -> list[TornadoBar]:
    """Assemble tornado bars from `{(factor, level): FOS}`, widest first.

    Requires a `("baseline", "base")` entry. A factor with only one non-base
    level gets that value on both sides -- one-sided bars are legitimate but
    they must be labelled as such in the caption, because a reader assumes
    symmetry.

    Ordering by span, not by |effect on FOS|, is deliberate: the tornado plot's
    job is to rank UNCERTAINTY contributions, and a factor that pushes FOS hard
    in one direction only still has a narrow span.
    """
    try:
        base = results[("baseline", "base")]
    except KeyError:
        raise ValueError(
            "tornado() needs a ('baseline','base') entry to centre on")

    by_factor: dict[str, dict[str, float]] = {}
    for (factor, level), fos in results.items():
        if factor == "baseline":
            continue
        by_factor.setdefault(factor, {})[level] = fos

    bars = []
    for factor, levels in by_factor.items():
        lo = levels.get("low", levels.get("high"))
        hi = levels.get("high", levels.get("low"))
        if lo is None or hi is None:
            continue
        bars.append(TornadoBar(factor=factor, low=lo, high=hi, base=base))

    return sorted(bars, key=lambda b: b.span, reverse=True)


# ---------------------------------------------------------------------------
# 4. Reproducibility and mesh convergence (Day 52)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeedStats:
    n: int
    mean: float
    sd: float

    def __str__(self) -> str:
        return f"FOS = {self.mean:.3f} +/- {self.sd:.3f} (1 SD, n = {self.n})"


def seed_statistics(fos_values) -> SeedStats:
    """Mean and SAMPLE standard deviation over random-seed repeats.

    Sample SD (n-1), not population: five seeds are a sample from the
    initialisation distribution, not the whole of it. At n = 5 the difference
    is 12% in the reported spread, which is the same order as the effects the
    tornado plot is trying to rank -- so it is not a pedantic distinction here.
    """
    vals = [float(v) for v in fos_values]
    if len(vals) < 2:
        raise ValueError(f"need at least 2 seeds, got {len(vals)}")
    return SeedStats(n=len(vals), mean=statistics.fmean(vals),
                     sd=statistics.stdev(vals))


def convergence_check(n_pde, fos_values, *, tol: float = 0.01):
    """Richardson-style check on the collocation-count refinement.

    Returns `(converged, rel_changes)` where `rel_changes[i]` is the relative
    change from refinement level i to i+1. Converged means the LAST change is
    within `tol`, which is the only one that speaks to the finest grid -- an
    early plateau followed by drift is not convergence, and averaging the
    changes would hide it.

    This is the PINN's analogue of mesh convergence, and it is the honest
    answer to the reproducibility question, but note what it does not test:
    collocation count and network capacity are separate knobs. A result that is
    flat in N_PDE can still be capacity-limited. If you have the budget for one
    extra arm, a width sweep at fixed N_PDE is worth more than a fifth N_PDE
    level.
    """
    n = np.asarray(n_pde, dtype=float)
    f = np.asarray([float(v) for v in fos_values], dtype=float)
    if n.size != f.size or n.size < 2:
        raise ValueError("need matched sequences of length >= 2")
    if not np.all(np.diff(n) > 0):
        raise ValueError("n_pde must be strictly increasing")

    rel = np.abs(np.diff(f)) / np.abs(f[:-1])
    return bool(rel[-1] <= tol), rel.tolist()


def coupling_effect(fos_two_way: float, fos_one_way: float) -> float:
    """Percent by which the one-way result overestimates FOS.

    Positive means one-way is UNCONSERVATIVE -- it reports a safer slope than
    the fully coupled solve does, which is the direction the framework is built
    to expose. A negative result is not a bug and must not be quietly reported
    as a magnitude; it would mean the feedback stiffens the response, and that
    would be the more interesting finding of the two.
    """
    return 100.0 * (fos_one_way - fos_two_way) / fos_two_way