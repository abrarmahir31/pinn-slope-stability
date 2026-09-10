"""
src/coupling.py — two-way coupling constitutive algebra. NO autograd.
=====================================================================

Layer position: beside `nondim.py`, not inside `residuals.py` or `mechanics.py`.
It takes ALREADY-DIFFERENTIATED quantities and returns constitutive
multipliers. Nothing here calls `torch.autograd.grad`, and nothing here
re-derives an equation that lives elsewhere.

Three coupling paths, all algebra:

  hydraulic <- mechanical :  Kozeny-Carman.  eps_v -> n -> K_s multiplier.
  mechanical <- hydraulic :  Bishop chi = Se, fed to `effective_stress_nd`.
  mechanical <- hydraulic :  bulk density rho_b = rho_dry + theta*rho_w.

chi and theta themselves come from `materials.Se` / `materials.theta`, which
are already torch-differentiable via `vg.py`. Nothing new is needed for them —
`materials.Se`'s docstring has said "also the Bishop chi in Step 3.3" since
Step 1. This module only clamps and assembles.

UNITS WARNING — read before using eps_v
---------------------------------------
Kozeny-Carman consumes PHYSICAL volumetric strain. With u* = u/U_ref and
x* = x/L_ref,

    eps_v_phys = (U_ref / L_ref) * eps_v_star = 0.01 * eps_v_star

at the current scales. Handing eps_v* to `porosity_from_strain` inflates the
feedback by a factor of 100. Use `mechanics.strain_star(..., physical=True)`.
This is the `ebf8f5c` bug class: an argument that arrives in the wrong units
and is never checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "KCConfig",
    "KC_OFF",
    "KC_ON",
    "KC_DEFAULT",
    "porosity_from_strain",
    "kozeny_carman_factor",
    "bishop_chi",
    "bulk_density_ratio",
]


# ---------------------------------------------------------------------------
# 1. Configuration — this IS the ablation switch
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KCConfig:
    """Porosity-strain feedback settings.

    enabled : False reproduces one-way coupling EXACTLY — the factor is
        identically 1.0 and carries no graph, so the Richards residual is
        bit-for-bit what it was before Step 3.3. That exactness is what makes
        the Step 6.2 ablation an ablation rather than a comparison of two
        approximations.

    ratio_min, ratio_max : bounds on n/n0, NOT on n. An absolute floor is
        wrong for Tm, whose n0 = 0.0221 — a floor of 1e-3 would still permit
        (n/n0)^3 to fall four orders. A ratio bound is the same physical
        statement for every stratum.

    n_max : absolute ceiling, because ((1-n0)/(1-n))^2 is singular at n = 1.
        Binds only where ratio_max would not.

    per_unit : optional {tag: bool} overriding `enabled` per stratum. Tm is
        87.3% of the domain by area, 2.2% porosity, and fracture-dominated
        (Finding 6) — applying a matrix-porosity law there is a decision, not
        a default. This is where you record it.

    known_tags : the stratum names `is_on` will accept. Defaults to the three
        Section 5 units. A tag outside this set raises rather than silently
        returning `enabled` — see `is_on`.
    """
    enabled: bool = False
    ratio_min: float = 0.5
    ratio_max: float = 1.5
    n_max: float = 0.95
    per_unit: dict | None = None
    known_tags: tuple = ("Mk", "Mk_d", "Tm")

    def is_on(self, tag: str | None = None) -> bool:
        """Is Kozeny-Carman feedback active for this stratum?

        UNKNOWN TAGS RAISE (Day 26). This used to fall through to `enabled`,
        so `KC_DEFAULT.is_on("TM")` returned True and a misspelled stratum
        name in `L_PDE` silently received the feedback that D-3.3.2 excluded
        it from -- with no error, no warning, and a Step 6.2 ablation whose
        two arms differ by a typo. `check_kc_range.py` found tag "XX" giving
        1.1104 at eps_v = 1e-2. Failing open on an identifier is never right
        when the identifier selects a physical law.

        `tag=None` is still allowed and still means "the global setting":
        that is the documented call for a domain-wide query.
        """
        if tag is None:
            return self.enabled
        if tag not in self.known_tags:
            raise KeyError(
                f"KCConfig.is_on: unknown stratum tag {tag!r}. Known: "
                f"{sorted(self.known_tags)}. Pass tag=None for the global "
                f"setting, or extend known_tags if the domain has grown.")
        if self.per_unit is not None and tag in self.per_unit:
            return bool(self.per_unit[tag])
        return self.enabled


KC_OFF = KCConfig(enabled=False)
KC_ON = KCConfig(enabled=True)

#: The configuration of record. Kozeny-Carman ON for the two marls, OFF for
#: Tm. See DECISIONS.md D-3.3.2 for the argument; the short version is that KC
#: is a matrix-porosity law and Tm's conductivity is fracture-controlled, so
#: applying it there would claim that compressing a limestone matrix closes
#: fractures the model does not represent -- possibly with the wrong sign.
#:
#: NOT a numerical decision. Measured against the gravity-equilibrated state,
#: Tm's KC factor at the 99th percentile of strain is 0.898, against 0.910 for
#: Mk_d: its low porosity makes it strain-SENSITIVE, but its 4.264e9 stiffness
#: means it barely strains, and the two effects very nearly cancel. Excluding
#: Tm therefore costs little and buys a defensible sentence.
#:
#: Flip it for the Step 6.2 sensitivity case:
#:     KCConfig(enabled=True)                        # KC everywhere
#:     KCConfig(enabled=True, per_unit={"Tm": False})  # this default
#:     KC_OFF                                        # one-way run
KC_DEFAULT = KCConfig(enabled=True, per_unit={"Tm": False})


# ---------------------------------------------------------------------------
# 2. Hydraulic <- mechanical
# ---------------------------------------------------------------------------
def porosity_from_strain(n0: float, eps_v: Tensor, cfg: KCConfig = KC_ON):
    """n = n0 + eps_v, clamped. `eps_v` is PHYSICAL strain (module header).

    Returns (n, clamped_mask). Tension-positive strain, so dilation raises n:
    opening the skeleton makes it more permeable.
    """
    n_raw = n0 + eps_v
    lo = cfg.ratio_min * n0
    hi = min(cfg.ratio_max * n0, cfg.n_max)
    if lo >= hi:
        raise ValueError(
            f"porosity clamp is empty for n0={n0:.4g}: lo={lo:.4g} >= hi={hi:.4g}. "
            "Check ratio_min/ratio_max/n_max."
        )
    n = n_raw.clamp(min=lo, max=hi)
    clamped = (n_raw < lo) | (n_raw > hi)
    return n, clamped


def kozeny_carman_factor(n0: float, eps_v: Tensor, cfg: KCConfig = KC_ON, *,
                         tag: str | None = None, return_diag: bool = False):
    """K_s(n) / K_s0  =  (n/n0)^3 * ((1-n0)/(1-n))^2.

    Disabled -> exactly ones, with no dependence on eps_v. `torch.ones_like`,
    not the float 1.0: returning a scalar would let the disabled path take a
    different route through the Richards residual than the enabled one, and
    the ablation would then be comparing two code paths, not two physics.
    """
    if not cfg.is_on(tag):
        f = torch.ones_like(eps_v)
        if return_diag:
            return f, {"enabled": False, "clamped_fraction": 0.0}
        return f

    n, clamped = porosity_from_strain(n0, eps_v, cfg)
    f = (n / n0) ** 3 * ((1.0 - n0) / (1.0 - n)) ** 2

    if return_diag:
        diag = {
            "enabled": True,
            "clamped_fraction": float(clamped.double().mean().detach()),
            "n_min": float(n.min().detach()),
            "n_max": float(n.max().detach()),
            "factor_min": float(f.min().detach()),
            "factor_max": float(f.max().detach()),
        }
        return f, diag
    return f


# ---------------------------------------------------------------------------
# 3. Mechanical <- hydraulic
# ---------------------------------------------------------------------------
def bishop_chi(se: Tensor) -> Tensor:
    """Bishop's parameter, chi = Se (effective saturation).

    A MODELLING CHOICE, and it should be named as one in the methods section.
    chi = Se is the standard closure when pore air is continuous and at
    atmospheric pressure — the same assumption that reduces two-phase flow to
    Richards, so it is at least self-consistent with the rest of the
    formulation. Khalili & Khabbaz's experimental fit has chi falling off
    faster than Se at low saturation. Carry it into the Step 6.2 sweep.

    Clamped to [0,1]: `vg.Se` is already bounded, but a network-produced psi
    can drive it outside during early training, and a negative chi flips the
    sign of the whole coupling term.
    """
    return se.clamp(0.0, 1.0)


def bulk_density_ratio(theta: Tensor, rho_dry: float, rho_b_ref: float,
                       rho_w: float = 1000.0) -> Tensor:
    """(rho_dry + theta*rho_w) / rho_b_ref — the `rho_ratio` argument that
    `nondim.mechanical_residual_nd` expects.

    theta is WATER CONTENT, not effective saturation. Use `materials.theta`,
    not `materials.Se`. For Mk the two differ by roughly 0.05 + 0.33*Se vs Se
    — a body force wrong by tens of percent, which is large enough to move the
    failure mechanism and small enough to look right on a contour plot.
    """
    return (rho_dry + theta * rho_w) / rho_b_ref