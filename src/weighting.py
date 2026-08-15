"""
weighting.py  --  Step 3.4: adaptive loss weighting for the coupled HM PINN
============================================================================
Isikdere lignite open-pit mine  (Ulusay et al., 2014, Engineering Geology)

Closes the CROSS-EQUATION imbalance documented in the Step 3.3 baseline note
(6cbd402): eight orders of magnitude in loss value between `bc_mech`
(1.459e-03) and `pde_richards` (1.217e-11).

D-W.0 -- THE NOTE MEASURED THE WRONG QUANTITY. MEASURE AGAIN FIRST.
--------------------------------------------------------------------
The baseline note tabulates L_i. The optimiser never sees L_i; it sees
grad_theta( sum_i w_i L_i ). For an MSE term L_i = (1/N) sum_k r_ik^2,

    grad L_i = (2/N) sum_k r_ik grad r_ik

so the gradient norm depends on r AND on grad r, and the relation between
g_i and L_i is not fixed. Write g_i/g_a = (L_i/L_a)^p. Two regimes, and this
domain contains both:

  p = 1   PREFACTOR-LIMITED. The residual carries a small multiplicative
          constant: r = Pi * f(theta), so grad r carries it too and
          g ~ Pi^2 ~ L. `pde_richards` is the candidate -- Pi_R_diff
          = 1.4948e-3 and Pi_R_grav = 1.5401e-3 against Pi_M_body = 3.0019,
          a ratio of ~1950 per residual, ~3.8e6 squared.

  p = 0.5 SATISFACTION-LIMITED. The residual is small because the current
          state nearly satisfies the constraint while grad r stays O(1), so
          g ~ sqrt(L). `ic_head` (2.13e-9) and `ic_disp` (2.77e-9) are here
          by construction -- the net is initialised at the physical state,
          psi* = psi_0* + 3e-3*net, u*,v* = 1e-3*net. So are the `base`,
          `far_field_f1` and `pit_floor` segments of `bc_mech`, for which
          u* = v* = 0 IS the solution at t = 0.

The two regimes give weights that differ by four orders on the same term:

    term            L_a/L_i    w at p=1    w at p=0.5
    pde_mech        8.98e+02   8.98e+02      3.00e+01
    bc              4.85e+03   4.85e+03      6.96e+01
    ic_disp         5.27e+05   5.27e+05      7.26e+02
    ic_head         6.84e+05   6.84e+05      8.27e+02
    pde_richards    1.20e+08   1.20e+08      1.10e+04

Neither can be inferred from the note's table. `w = L_a/L_i` and
`w = sqrt(L_a/L_i)` are BOUNDS, not a scheme. The first action of Step 3.4
is therefore `scripts/grad_norm_table.py`, which measures g_i directly; the
balancer below uses g_i and never touches L_i.

`regime_exponents()` reports the fitted p per term. That table earns a
paragraph in the methods section on its own: it separates "this term is
weakly scaled" from "this term is nearly satisfied at t = 0", and only the
first is a scaling defect that the weights should be fixing.

D-W.0b -- and measure at a state that is not t = 0
---------------------------------------------------
The baseline is taken at a near-equilibrium initialisation. `pde_richards`
is small there partly because the initial state IS a Richards solution --
test_initial_equilibrium.py pins exactly that. Weights fitted only at t = 0
are fitted to the satisfaction-limited regime and will be wrong once
drawdown starts. Run the measurement twice: at the initialisation, and again
after ~500 Adam steps with the drawdown BC live. If p moves, the adaptive
scheme is doing necessary work and a static weight set would not have been
enough -- which is itself the argument for Step 3.4 existing at all.

D-W.1 -- balance gradient norms, not loss ratios
------------------------------------------------
Roadmap Step 3.4 specifies GradNorm (Chen et al., 2018), whose relative
training rate is r~_i = L_i(t) / L_i(0). Four of the seven terms have
L_i(0) at round-off by construction (ic_head, ic_disp, interface, and three
of the six bc_mech segments), so that denominator is meaningless or
overflowing. The adopted update is the learning-rate-annealing rule of
Wang, Teng & Perdikaris (2021), which uses only current gradient norms:

    w^_i  =  g_anchor / g_i

GradNorm's alpha-exponent rate term is implemented but OFF by default
(`alpha=0.0`). Turn it on only once every term has a nonzero baseline --
realistically after the first drawdown step, not before.

D-W.2 -- anchor on bc_mech; every other weight climbs to meet it
----------------------------------------------------------------
The note establishes that bc_mech is large because the cut face is genuinely
out of equilibrium at t = 0 (cut_face 5.70e-03, 13x natural_ground) and that
suppressing it would suppress the constraint on the failure surface. So
w_bc_mech is PINNED AT 1.0 and never updated; balance is reached by raising
the others. The scheme is then monotone in the safe direction -- it can
never quietly turn the mechanical boundary off.

    Rejected: target = geometric mean of the active norms. It is the
    scale-free centre and it keeps the total loss scale fixed, which is
    tidier. But on this spread it puts bc_mech BELOW 1 -- it weights the
    cut-face constraint down for the sole reason that it is large, the exact
    move the note forbids. `target="geomean"` is implemented and tested so
    the comparison can be run and reported, but it is not the default and
    should not be adopted without re-arguing D-W.2.

Two costs of anchoring, stated because both will look like bugs:
  * the total loss RISES. At p = 1 it goes 1.46e-3 -> 8.75e-3 (each term
    contributes 1.459e-3 by construction); at p = 0.5, 1.46e-3 -> 1.53e-3.
    The Step 4.1 learning-rate schedule is calibrated against the weighted
    loss, not the unit-weight one.
  * the loss-VALUE spread does not close. Under exact gradient balance at
    p = 0.5, w_i L_i = sqrt(L_a L_i): the 8-order spread halves to 4 and
    stays there. That is correct. Anyone monitoring only the total loss will
    conclude the weighting did nothing; the quantity to monitor is w_i g_i,
    which `report()` prints, along with its spread.

D-W.3 -- log-space EMA on a CORRECTION FACTOR, hard clip
--------------------------------------------------------
Each weight is factored as

    w_i  =  w0_i * c_i

with w0_i a static seed fixed once (D-W.7) and c_i the adaptive correction,
updated multiplicatively:

    log10 c_i  <-  (1 - lam) log10 c_i  +  lam log10 (w^_i / w0_i)

Log space because a linear EMA at lam = 0.1 needs ~1e5 updates to travel
from 1 to 1e4, i.e. it never arrives. The clip is on log10 c, not on log10
w, so `log_clip = (-3, +3)` means "the adaptive scheme may not disagree with
the physics-derived seed by more than three orders" -- a statement about the
scheme. Clipping log10 w instead would have meant "no weight may exceed
1e6", which is a statement about nothing, and would silently cap
pde_richards below the 1.2e8 that the p = 1 regime demands. A correction
pressed against the clip is a diagnostic, not a success; `report()` flags it
CLIPPED, and a persistent one means the seed is wrong, not the clip.

D-W.7 -- seed the weights from the Pi groups, then correct adaptively
---------------------------------------------------------------------
Most of the cross-equation gap is a KNOWN constant. The mechanical residual
was scaled so Pi_M_body = 3.0019 makes it O(1); the Richards residual was
left at normalise="none" with Pi_R_grav = 1.5401e-3, on the sound D-S.4
argument that no term dominates inside that equation. The ratio 1949 is not
an emergent property of training -- it is arithmetic available before the
first epoch, and squared (3.8e6) it is the p = 1 weight for pde_richards.

Handing that to GradNorm to discover costs the whole warm-up budget, which
is the failure the roadmap warns about for skipped non-dimensionalisation.
So `PI_SEED` supplies it statically and the balancer corrects the remaining
factor, which is O(10-100) and lives comfortably inside the clip. This also
keeps the two mechanisms separable in the methods section: the seed is
derived, the correction is measured, and `report()` prints both.

The seed must NOT be implemented by dividing the Richards residual -- that
is what D-S.4 rejected, and rightly: rescaling residual and gradient
together only relabels the imbalance. A loss weight multiplies the squared
residual and does change the gradient the optimiser sees. The distinction is
the whole reason the fix belongs here and not in residuals.py.

D-W.4 -- inactive terms hold their weight, they are not amplified
-----------------------------------------------------------------
`interface` at 1.238e-32 is zero because the initial state satisfies flux
continuity exactly. It is an UNEXERCISED term, not a small one; it will grow
once drawdown reaches the Mk/Tm contact. Any 1/g rule applied to it returns
~1e14 (p = 0.5) or ~1e29 (p = 1) and destroys training on the first update.
Terms whose gradient norm falls below `activity_floor * g_max` are frozen at
their current weight and excluded from the target, and rejoin the pool
automatically when they activate. `report()` names them, so a term frozen
for the wrong reason is visible rather than silent.

D-W.5 -- per-stratum weighting is DEFERRED; per-stratum reporting is NOT
------------------------------------------------------------------------
The note shows Tm at 48x Mk in pde_mech and 53x below it in pde_richards
(Tm is 87.3% of the domain, 20-100x stiffer, 3300x more conductive). A
single global weight cannot express that. Deferring is still right:
splitting 7 terms into ~13 before the global scheme has been shown to fail
adds hyperparameters with no evidence for them. What is NOT deferred is
measurement -- pass `subkeys={"pde_mech": ("Mk", "Mk_d", "Tm"), ...}` and the
balancer logs per-stratum gradient norms alongside the global ones while
acting on nothing. Step 4.1 then decides on data, and the Step 6.2 two-way-
coupling ablation is reported per-stratum as the note requires.

Escalation if the global scheme stalls: self-adaptive pointwise weights
(McClenny & Braga-Neto, SA-PINN) subsume per-stratum weighting without
anyone hand-designing the groups; NTK weighting (Wang, Yu & Perdikaris 2022)
is the more principled, more expensive alternative named in the roadmap.

D-W.6 -- weighting must not move the zero
------------------------------------------
Every w_i multiplies a residual that is exactly zero at the initial state,
so no weight vector can move it; tests/test_initial_equilibrium.py is
unaffected by this module and must stay unaffected. What CAN break is the
balancer returning NaN/Inf from a 0/0 and poisoning theta on the next step.
That is what tests/test_weighting.py guards.

OPEN -- for docs/open_items.md
------------------------------
- [ ] `interface` carries w0 = 3.8e6 while frozen, so it enters the objective
      at full seed weight the instant it activates. The balancer corrects it
      within a few updates, but the first weighted step is a shock. Consider a
      ramp over ~10 updates on transition from frozen to active. Not urgent:
      the term activates when drawdown reaches the Mk/Tm contact, which is
      well after the Adam warm-up.
- [ ] The seed applies (Pi_M_body/Pi_R_grav)^2 to `pde_richards`, using the
      GRAVITY group. Pi_R_diff = 1.4948e-3 differs from Pi_R_grav = 1.5401e-3
      by 3%, which is inside the noise here, but the choice should be recorded
      rather than inherited. Note the ratio L/H = 5.67 between them is fixed by
      geometry (open_items, Step 1.6 reopened 4 Aug), so no T_ref makes both
      O(1) and the seed cannot be made exact for both terms.
- [ ] Once the Step 4.1 run exists, check whether the fitted correction c ever
      settles. If c drifts monotonically for 10 000+ epochs the anchor is
      wrong, not the correction.
- [ ] E_ref = 1e9 vs the Phase-1 decision to use 100 MPa is still open
      (open_items, Step 3.2 6 Aug) and it moves sig_ref-derived quantities,
      hence Pi_M_body, hence PI_SEED. Settle it before any long run, or the
      seed is measured against a scaling that will change under it.

Author: <thesis>  |  Phase 3, Step 3.4
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. Gradient-norm measurement -- run this BEFORE choosing any weights
# ---------------------------------------------------------------------------
def grad_norm(loss: Tensor, params: Sequence[torch.nn.Parameter],
              retain: bool = True) -> float:
    """|| d loss / d theta ||_2 over the flattened parameter vector.

    Unused parameters contribute zero rather than raising. Returns 0.0 for a
    loss with no graph -- a term that is identically zero by construction.
    The caller decides what that means; see `LossBalancer._active`.
    """
    if not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=retain,
                                allow_unused=True, create_graph=False)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum())
    return math.sqrt(total)


def grad_norm_table(losses: Mapping[str, Tensor],
                    params: Sequence[torch.nn.Parameter]) -> Dict[str, float]:
    """Gradient norms for every term. One backward pass per term.

    Cost: n_terms backward passes. At n = 7 and one call per 100 optimiser
    steps this is ~7% overhead on a 2x64 network. Do not call it every step.
    """
    return {k: grad_norm(v, params) for k, v in losses.items()}


def regime_exponents(snap_a: Tuple[Mapping[str, float], Mapping[str, float]],
                     snap_b: Tuple[Mapping[str, float], Mapping[str, float]]
                     ) -> Dict[str, float]:
    """Fitted p in  g ~ L^p , per term, from TWO states. See D-W.0 / D-W.0b.

    Each snapshot is (losses, norms). p_i = dlog g_i / dlog L_i measured on
    the same term at two different network states:

        p ~ 1.0  prefactor-limited -- the residual carries a small constant,
                 so grad r carries it too. A scaling defect; the weights
                 should fix it.
        p ~ 0.5  satisfaction-limited -- the residual is small because the
                 state nearly satisfies it, while grad r stays O(1). NOT a
                 defect. Weighting it to p = 1 levels over-drives a
                 constraint that is already met and can destabilise Adam.

    ACROSS terms this cannot be fitted: g_i = k_i L_i^p with a per-term
    constant k_i that folds into the exponent and biases it by whatever the
    residuals happen to look like. The first draft of this function did the
    cross-term version and reported p = 0.16 for a term constructed to be
    exactly 0.5. Two states, one term, is the only estimator that works.

    In practice: snapshot at the initialisation, run ~500 Adam steps with the
    drawdown BC live, snapshot again. Terms outside [0.3, 1.2] have structure
    neither model captures and should be understood before being weighted.
    """
    (La, ga), (Lb, gb) = snap_a, snap_b
    out: Dict[str, float] = {}
    for k in La:
        if k not in Lb or k not in ga or k not in gb:
            continue
        l0, l1, g0, g1 = La[k], Lb[k], ga[k], gb[k]
        if min(l0, l1, g0, g1) <= 0:
            continue
        den = math.log(l1 / l0)
        if abs(den) < 1e-9:            # term did not move; no information
            continue
        out[k] = math.log(g1 / g0) / den
    return out


def format_table(norms: Mapping[str, float],
                 anchor: str = "bc_mech",
                 p: Optional[Mapping[str, float]] = None) -> str:
    """Rank-ordered gradient-norm table, in the format of the baseline note.

    `p` is optional and comes from `regime_exponents` on two snapshots; pass
    it once the second measurement exists.
    """
    if not norms:
        return "(no terms)"
    items = sorted(norms.items(), key=lambda kv: -kv[1])
    top = norms.get(anchor) or items[0][1]
    head = f"{'term':<16}{'||grad L||':>14}{'g_a/g_i':>13}{'w^ = g_a/g_i':>15}"
    if p:
        head += f"{'p':>7}  regime"
    lines = [head]
    for k, g in items:
        if g <= 0:
            lines.append(f"{k:<16}{g:>14.4e}{'inactive':>13}{'held':>15}")
            continue
        row = f"{k:<16}{g:>14.4e}{top / g:>13.3e}{top / g:>15.3e}"
        if p:
            pk = p.get(k)
            if pk is None:
                row += f"{'-':>7}  (not measured)"
            else:
                tag = ("prefactor" if pk > 0.75 else
                       "satisfaction" if pk > 0.35 else "UNEXPLAINED")
                row += f"{pk:>7.2f}  {tag}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. The balancer
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# D-W.7 -- static seed from the dimensionless groups
# ---------------------------------------------------------------------------
# Values from src/nondim.py at 6cbd402. Imported lazily in `pi_seed()` so this
# module stays importable in a bare test environment; the literals below are
# the fallback and are asserted against nondim.SCALES in the test suite.
_PI_M_BODY = 3.0019       # rho_b_ref g L_ref / sig_ref     (mechanical, O(1))
_PI_R_GRAV = 1.5401e-3    # K_ref T_ref / (L_ref dtheta_ref)
_PI_R_DIFF = 1.4948e-3    # K_ref T_ref H_ref / (L_ref^2 dtheta_ref)


def pi_seed(pi_m: float = _PI_M_BODY, pi_r: float = _PI_R_GRAV
            ) -> Dict[str, float]:
    """Static weights w0 from the ratio the scaling decisions left open.

    The Richards residual carries Pi_R ~ 1.54e-3 where the mechanical one
    carries Pi_M_body = 3.0019. Both are correct choices (D-S.4 closed the
    Richards scaling as normalise="none"; Pi_M_body was chosen to make the
    mechanical residual O(1)); the CROSS-equation ratio is the leftover, and
    it is (Pi_M/Pi_R)^2 in the squared loss.

    Only the Richards terms are seeded. The mechanical PDE and all the
    boundary/initial terms live on the mechanical scale already and start at
    w0 = 1, so any weight they end up with is a measured correction and can
    be reported as such.
    """
    r = (pi_m / pi_r) ** 2
    return {"bc_mech": 1.0, "pde_mech": 1.0, "bc": 1.0,
            "ic_disp": 1.0, "ic_head": 1.0,
            "pde_richards": r, "interface": r}


PI_SEED = pi_seed()


@dataclass
class BalancerConfig:
    anchor: str = "bc_mech"          # D-W.2: pinned at 1.0, never updated
    target: str = "anchor"           # "anchor" (D-W.2) | "geomean" | "mean"
    lam: float = 0.1                 # log-space EMA rate on the correction
    every: int = 100                 # optimiser steps between updates
    alpha: float = 0.0               # GradNorm rate exponent; 0 = off (D-W.1)
    log_clip: Tuple[float, float] = (-3.0, 3.0)     # log10 bounds on c, D-W.3
    activity_floor: float = 1e-12    # relative to the largest grad norm
    warmup: int = 500                # steps before the first update
    seed: Optional[Mapping[str, float]] = None      # w0; None -> PI_SEED


class LossBalancer:
    """Adaptive weights for the monolithic HM loss.

    Usage inside the Step 4.1 Adam loop:

        bal = LossBalancer(TERMS, net.parameters())
        for step in range(n_epochs):
            losses = loss_fn(...)               # dict[str, 0-d Tensor]
            bal.maybe_update(step, losses)      # no-op off-cadence
            total = bal.total(losses)
            opt.zero_grad(); total.backward(); opt.step()

    `maybe_update` must be called BEFORE `total.backward()` -- it needs the
    graph intact and does its own autograd.grad calls with retain_graph=True.

    Weights are plain Python floats: hyperparameters of the current step, not
    trainable quantities. No gradient flows through w_i, which is what stops
    the optimiser from minimising the objective by shrinking w instead of by
    satisfying physics.
    """

    def __init__(self, term_names: Iterable[str],
                 params: Iterable[torch.nn.Parameter],
                 cfg: Optional[BalancerConfig] = None,
                 subkeys: Optional[Mapping[str, Sequence[str]]] = None):
        self.cfg = cfg or BalancerConfig()
        self.params = list(params)
        self.terms = list(term_names)
        if self.cfg.anchor not in self.terms:
            raise ValueError(f"anchor {self.cfg.anchor!r} not among terms "
                             f"{self.terms}")
        seed = dict(PI_SEED if self.cfg.seed is None else self.cfg.seed)
        self.w0: Dict[str, float] = {k: float(seed.get(k, 1.0))
                                     for k in self.terms}
        self.w0[self.cfg.anchor] = 1.0            # D-W.2: anchor defines 1
        self.c: Dict[str, float] = {k: 1.0 for k in self.terms}
        self.subkeys = dict(subkeys or {})        # D-W.5: report only
        self.history: list = []
        self.snapshots: list = []      # (losses, norms) pairs for D-W.0b
        self._frozen: set = set()
        self._clipped: set = set()
        self._last_norms: Dict[str, float] = {}
        self._n_updates = 0

    # -- weights -----------------------------------------------------------
    @property
    def w(self) -> Dict[str, float]:
        """Effective weights, w_i = w0_i * c_i. Read-only by construction:
        the seed is a physics decision (D-W.7) and the correction is measured
        (D-W.1); neither should be poked at from a training script."""
        return {k: self.w0[k] * self.c[k] for k in self.terms}

    # -- internals ---------------------------------------------------------
    def _active(self, norms: Mapping[str, float]) -> Dict[str, float]:
        """Terms carrying a usable gradient. D-W.4."""
        gmax = max(norms.values(), default=0.0)
        if gmax <= 0.0:
            return {}
        floor = self.cfg.activity_floor * gmax
        return {k: g for k, g in norms.items() if g > floor}

    def _target(self, active: Mapping[str, float]) -> float:
        mode = self.cfg.target
        if mode == "anchor":
            g = active.get(self.cfg.anchor)
            if g is None:
                raise RuntimeError(
                    f"target='anchor' but {self.cfg.anchor} is inactive "
                    f"(grad norm {self._last_norms.get(self.cfg.anchor, 0):.3e})."
                    " Either the anchor is wrong or the state is degenerate.")
            return g
        vals = list(active.values())
        if mode == "geomean":
            return math.exp(sum(math.log(v) for v in vals) / len(vals))
        if mode == "mean":
            return sum(vals) / len(vals)
        raise ValueError(f"target={mode!r} not in "
                         "{'anchor', 'geomean', 'mean'}")

    # -- public API --------------------------------------------------------
    def update(self, losses: Mapping[str, Tensor]) -> Dict[str, float]:
        """One weight update. Returns the new weights."""
        norms = grad_norm_table({k: losses[k] for k in self.terms
                                 if k in losses}, self.params)
        self._last_norms = dict(norms)
        self.snapshots.append(({k: float(losses[k].detach()) for k in norms},
                               dict(norms)))
        if len(self.snapshots) > 2:    # keep first and latest only
            self.snapshots = [self.snapshots[0], self.snapshots[-1]]
        active = self._active(norms)
        self._frozen = set(self.terms) - set(active)
        if len(active) < 2:
            # nothing to balance against; leave the weights alone rather than
            # inventing a target from one number.
            return dict(self.w)

        tgt = self._target(active)
        lo, hi = self.cfg.log_clip
        self._clipped = set()

        for k, g in active.items():
            if k == self.cfg.anchor:
                continue                       # D-W.2: pinned
            w_hat = tgt / g
            if self.cfg.alpha:                 # GradNorm rate term, opt-in
                w_hat *= self._rate(k, losses) ** self.cfg.alpha
            c_hat = w_hat / self.w0[k]         # D-W.7: correct the seed only
            log_c = ((1.0 - self.cfg.lam) * math.log10(max(self.c[k], 1e-300))
                     + self.cfg.lam * math.log10(max(c_hat, 1e-300)))
            if log_c <= lo or log_c >= hi:
                self._clipped.add(k)
            self.c[k] = 10.0 ** min(max(log_c, lo), hi)

        self.c[self.cfg.anchor] = 1.0
        self._n_updates += 1

        # D-W.5: measure the per-stratum split, act on nothing.
        sub: Dict[str, float] = {}
        for parent, keys in self.subkeys.items():
            for kk in keys:
                name = f"{parent}::{kk}"
                if name in losses:
                    sub[name] = grad_norm(losses[name], self.params)
        self.history.append({"norms": dict(norms), "sub": sub,
                             "w": dict(self.w), "c": dict(self.c),
                             "target": tgt})
        return dict(self.w)

    def maybe_update(self, step: int,
                     losses: Mapping[str, Tensor]) -> Dict[str, float]:
        """Update on cadence; no-op otherwise. Cheap to call every step."""
        if step >= self.cfg.warmup and step % self.cfg.every == 0:
            return self.update(losses)
        return dict(self.w)

    def total(self, losses: Mapping[str, Tensor]) -> Tensor:
        """sum_i w_i L_i, w detached. Missing terms are an error, not a
        default -- a silently dropped physics term is exactly the failure the
        roadmap warns about in Step 3.2."""
        missing = [k for k in self.terms if k not in losses]
        if missing:
            raise KeyError(f"loss dict is missing {missing}; refusing to "
                           "assemble a partial objective")
        w = self.w
        out = None
        for k in self.terms:
            piece = w[k] * losses[k]
            out = piece if out is None else out + piece
        return out

    def _rate(self, k: str, losses: Mapping[str, Tensor]) -> float:
        """GradNorm relative inverse training rate. See D-W.1 for why this is
        off by default: L_i(0) is round-off for four of the seven terms."""
        if not hasattr(self, "_L0"):
            self._L0 = {kk: max(float(losses[kk]), 1e-300) for kk in self.terms}
        r = float(losses[k]) / self._L0[k]
        mean_r = sum(float(losses[kk]) / self._L0[kk]
                     for kk in self.terms) / len(self.terms)
        return r / mean_r if mean_r > 0 else 1.0

    # -- diagnostics -------------------------------------------------------
    def report(self) -> str:
        w = self.w
        lines = [f"LossBalancer  updates={self._n_updates}  "
                 f"anchor={self.cfg.anchor}  target={self.cfg.target}",
                 f"{'term':<16}{'||grad L||':>13}{'w0 (seed)':>12}"
                 f"{'c (adapt)':>12}{'w':>12}{'w*||grad L||':>15}  flag"]
        for k in self.terms:
            g = self._last_norms.get(k, float("nan"))
            flag = ("FROZEN" if k in self._frozen else
                    "CLIPPED" if k in self._clipped else
                    "anchor" if k == self.cfg.anchor else "")
            lines.append(f"{k:<16}{g:>13.4e}{self.w0[k]:>12.3e}"
                         f"{self.c[k]:>12.3e}{w[k]:>12.3e}"
                         f"{w[k] * g:>15.4e}  {flag}")
        live = [w[k] * self._last_norms.get(k, 0.0) for k in self.terms
                if self._last_norms.get(k, 0.0) > 0]
        if live:
            lines.append(f"  weighted-gradient spread: "
                         f"{max(live) / min(live):.3e}   (target 1)")
        if self.snapshots:
            p = regime_exponents(self.snapshots[0], self.snapshots[-1])
            if p:
                lines.append("  regime exponents p "
                             "(1 = prefactor, 0.5 = satisfaction):")
                lines.append("    " + "  ".join(f"{k}={v:.2f}"
                                                for k, v in p.items()))
        if self.history and self.history[-1]["sub"]:
            lines.append("  per-stratum (reported, not weighted):")
            for k, g in sorted(self.history[-1]["sub"].items(),
                               key=lambda kv: -kv[1]):
                lines.append(f"    {k:<28}{g:>13.4e}")
        return "\n".join(lines)

    def state_dict(self) -> dict:
        """Both halves. Resuming with c = 1 after 40 000 epochs silently
        restarts the balancing; resuming with a different w0 silently changes
        the objective, so the seed is checkpointed too and checked on load."""
        return {"w0": dict(self.w0), "c": dict(self.c),
                "w": dict(self.w), "n_updates": self._n_updates,
                "cfg": dict(self.cfg.__dict__)}

    def load_state_dict(self, d: Mapping) -> None:
        w0 = d.get("w0")
        if w0 is not None:
            drift = [k for k in self.terms
                     if abs(math.log10(max(w0.get(k, 1.0), 1e-300)
                                       / max(self.w0[k], 1e-300))) > 1e-9]
            if drift:
                raise ValueError(
                    f"checkpoint seed differs from current PI_SEED on {drift}; "
                    "the objective is not the same one that was trained. "
                    "Pass cfg.seed explicitly if this is intentional.")
        self.c.update(d["c"])
        self._n_updates = int(d.get("n_updates", 0))


# ---------------------------------------------------------------------------
# 3. Term registry and the two weight bounds
# ---------------------------------------------------------------------------
TERMS = ("bc_mech", "pde_mech", "bc", "ic_disp", "ic_head",
         "pde_richards", "interface")

#: Measured loss values, Step 3.3 baseline (6cbd402), unit weights, N = 300.
BASELINE_LOSSES = {
    "bc_mech":      1.459e-03,
    "pde_mech":     1.624e-06,
    "bc":           3.010e-07,
    "ic_disp":      2.769e-09,
    "ic_head":      2.134e-09,
    "pde_richards": 1.217e-11,
    "interface":    1.238e-32,
}

#: bc_mech segment split, same measurement. Reported, never weighted.
BASELINE_BC_MECH_SEGMENTS = {
    "cut_face":       5.695e-03, "bench":        4.316e-04,
    "natural_ground": 1.535e-04, "base":         7.833e-08,
    "far_field_f1":   2.981e-09, "pit_floor":    1.151e-09,
}

#: Per-stratum PDE residuals, same measurement. D-W.5.
BASELINE_PER_STRATUM = {
    "pde_mech":     {"Mk": 3.861e-08, "Mk_d": 1.874e-08, "Tm": 1.863e-06},
    "pde_richards": {"Mk": 1.477e-10, "Mk_d": 5.316e-11, "Tm": 2.762e-12},
}


def weight_bounds(losses: Mapping[str, float] = BASELINE_LOSSES,
                  anchor: str = "bc_mech",
                  exclude: Sequence[str] = ("interface",)
                  ) -> Dict[str, Tuple[float, float]]:
    """(w at p=0.5, w at p=1) per term. ORIENTATION ONLY -- NOT A BRACKET.

    It is tempting to treat [sqrt(L_a/L_i), L_a/L_i] as a range the measured
    w^ = g_a/g_i must fall inside, and skip the measurement. It is not:
    g_i = k_i * L_i^p carries a per-term constant k_i, and the loss table
    cannot see k_i. A term can and does measure OUTSIDE the pair on either
    side (test_weight_bounds_are_not_actually_bounds pins a case at w^ = 3.9
    against a nominal [7.0e1, 4.8e3]).

    The pair is here for one purpose: to show that the note's loss table
    leaves the weights undetermined across four orders even under the two
    most charitable models, so the gradient measurement is not optional.

    `interface` is excluded, not bracketed: at 1.238e-32 the pair is
    [3.4e14, 1.2e29] and neither end is a weight. See D-W.4.
    """
    la = losses[anchor]
    out: Dict[str, Tuple[float, float]] = {}
    for k, v in losses.items():
        if k in exclude or v <= 0:
            continue
        r = la / v
        out[k] = (math.sqrt(r), r)
    return out


if __name__ == "__main__":
    print("Step 3.4 -- what the 6cbd402 LOSS table can and cannot tell you\n")
    print("The two columns below are the same weight under the two regimes of")
    print("D-W.0. They differ by four orders. Neither is the answer, and the")
    print("measurement is not guaranteed to land between them -- run")
    print("scripts/grad_norm_table.py.\n")
    print(f"{'term':<16}{'L':>13}{'L_a/L':>12}{'w @ p=0.5':>12}"
          f"{'w @ p=1':>12}{'w*L @ p=1':>13}")
    b = weight_bounds()
    la = BASELINE_LOSSES["bc_mech"]
    for k, v in BASELINE_LOSSES.items():
        if k not in b:
            print(f"{k:<16}{v:>13.3e}      INACTIVE (D-W.4) -- weight held")
            continue
        lo, hi = b[k]
        print(f"{k:<16}{v:>13.3e}{la / v:>12.3e}{lo:>12.3e}{hi:>12.3e}"
              f"{hi * v:>13.3e}")
    print(f"\ntotal loss   unit weights : {sum(BASELINE_LOSSES.values()):.4e}")
    print("             p = 0.5      : "
          f"{sum(math.sqrt(la * v) for k, v in BASELINE_LOSSES.items() if k in b):.4e}")
    print(f"             p = 1        : {la * len(b):.4e}")
    print("\nD-W.7 static seed from the Pi groups "
          f"((Pi_M_body/Pi_R_grav)^2 = {PI_SEED['pde_richards']:.3e}):")
    for k, v in PI_SEED.items():
        print(f"  w0[{k}] = {v:.4e}")
    print("\nThe adaptive correction c only has to cover what the seed misses.")