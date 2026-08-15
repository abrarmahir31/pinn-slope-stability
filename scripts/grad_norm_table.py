"""
grad_norm_table.py -- Step 3.4, measurement 1. RUN THIS BEFORE SETTING WEIGHTS.

    PYTHONPATH=. python scripts/grad_norm_table.py --selftest
    PYTHONPATH=. python scripts/grad_norm_table.py
    PYTHONPATH=. python scripts/grad_norm_table.py --adam-steps 500 --json out.json

Produces the table docs/step34_baseline.md is missing: || grad_theta L_i || for
every loss term, at the state the baseline note measured, plus the regime
exponent p per term from a second state.

The note tabulates L_i, which is not what the weights are computed from
(D-W.0). Under the two plausible models -- the residual is small because it
carries a small Pi group (p = 1), or small because the state nearly satisfies
it (p = 0.5) -- the SAME term wants weights four orders apart. And because
g_i = k_i L_i^p carries a per-term constant the loss table cannot see, the
measured value is not even guaranteed to fall between the two.

WHAT TO LOOK FOR
----------------
1. `w^ = g_a/g_i` is what the balancer converges to at lam = 1. Compare it to
   the `c = w^/w0` column: if every c sits inside the default +/-3 orders, the
   D-W.7 seed is doing its job.

2. The `p` column (needs --adam-steps). p ~ 1 means the term is prefactor-
   limited and the weight fixes a real scaling defect. p ~ 0.5 means it is
   merely nearly satisfied AT THIS STATE, and a large weight over-drives a
   constraint already met. `ic_head`, `ic_disp` and the constrained bc
   segments should land near 0.5 by construction. If `pde_richards` also
   lands near 0.5, the eight-order gap in the note is mostly an artefact of
   measuring at t = 0 on a near-equilibrium init and the weighting problem is
   smaller than it looks. Good news, but check it rather than assume it.

3. `interface` must report inactive. A finite gradient norm there at t = 0
   means the initial state no longer satisfies flux continuity exactly, and
   tests/test_initial_equilibrium.py should have caught it first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

from src.weighting import (PI_SEED, TERMS, BalancerConfig, format_table,
                           grad_norm_table, regime_exponents, weight_bounds)

# ===========================================================================
# EDIT ONLY THIS BLOCK. Everything below it is generic.
#
# Two functions. Both must reuse the Step 3.3 builder rather than re-rolling
# one -- a different initialisation gives a different table, and an unscaled
# random net emits psi* ~ 1, i.e. 165 m of suction, at which K_r is
# numerically zero and every flux term collapses twelve orders. That regime
# already fooled the Step 3.3b tests once.
# ===========================================================================

def build(seed: int = 20250812, N: int = 300):
    import torch
    from src.loss import ic_targets
    from src.materials import load_materials
    from src.sampling import (sample_boundary, sample_interfaces,
                              sample_interior)
    from src.sigma0 import attach_sigma0
    from src.nondim import SCALES

    DTYPE = torch.float64

    class Step33Net(torch.nn.Module):
        """2x64 tanh, initialised near the physical state (6cbd402).

        psi* = psi_0* + 3e-3*net,  u*,v* = 1e-3*net.
        psi_0* is the Step 2.3 IC: psi = -(z - z_wt), z_wt = 197.0 m a.s.l.,
        non-dimensionalised by H_ref. Without this offset the raw net emits
        psi* ~ 1 = 165 m suction and the whole table is meaningless.
        """
        Z_WT = 197.0

        def __init__(self, seed=seed, s=SCALES):
            super().__init__()
            torch.manual_seed(seed)
            self.s = s
            self.net = torch.nn.Sequential(
                torch.nn.Linear(3, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, 64), torch.nn.Tanh(),
                torch.nn.Linear(64, 3),
            ).to(DTYPE)

        def forward(self, x, z, t):
            raw = self.net(torch.cat([x, z, t], dim=1))
            psi0 = -(z * self.s.L_ref - self.Z_WT) / self.s.H_ref
            return torch.cat([psi0 + 3e-3 * raw[:, 0:1],
                              1e-3 * raw[:, 1:3]], dim=1)

    coll = sample_interior(N, seed)
    attach_sigma0(coll)
    bcs = sample_boundary(N, seed, 30.0)
    for c in bcs.values():
        attach_sigma0(c)

    loss_fn = dict(bcs=bcs, ic=ic_targets(),
                   ifaces=sample_interfaces(N, seed), mats=load_materials())
    return Step33Net(), loss_fn, coll


def losses_at(net, loss_fn, colloc):
    from src.coupling import KC_DEFAULT
    from src.loss import L_BC, L_BC_mech, L_IC, L_PDE, L_PDE_mech, L_interface

    b, mats = loss_fn, loss_fn["mats"]
    out = {
        "pde_richards": L_PDE(net, colloc, mats, kc=KC_DEFAULT)[0],
        "pde_mech":     L_PDE_mech(net, colloc, mats)[0],
        "bc":           L_BC(net, b["bcs"], mats)[0],
        "bc_mech":      L_BC_mech(net, b["bcs"], mats)[0],
        "interface":    L_interface(net, b["ifaces"], mats)[0],
    }
    _, ic_parts = L_IC(net, *b["ic"], keep_graph=True)   # needs the L_IC edit
    out["ic_head"], out["ic_disp"] = ic_parts["ic_head"], ic_parts["ic_disp"]
    return out

# ===========================================================================
# END EDIT BLOCK
# ===========================================================================


def _selftest_build(seed: int = 0):
    """Synthetic stand-in reproducing the 6cbd402 loss MAGNITUDES and both
    regimes of D-W.0. Not physics -- it exists so the plumbing (grad norms,
    seed comparison, regime fit, JSON) can be verified before the real loss
    is wired, and so a later failure is unambiguously in the loss, not here.
    """
    from src.weighting import BASELINE_LOSSES
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(3, 64, dtype=torch.float64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64, dtype=torch.float64), torch.nn.Tanh(),
        torch.nn.Linear(64, 3, dtype=torch.float64))
    colloc = torch.randn(300, 3, dtype=torch.float64)
    regime = {"bc_mech": 1.0, "pde_mech": 1.0, "pde_richards": 1.0,
              "bc": 0.5, "ic_disp": 0.5, "ic_head": 0.5}

    # scale constants fixed ONCE, from the initial state, so that later
    # theta changes actually move L. Recomputing them per call would pin every
    # term at its target and the regime fit would have nothing to fit.
    with torch.no_grad():
        f0 = net(colloc)[:, 0]
        m0 = float((f0 ** 2).mean())
        a0 = float(f0.mean())
    const = {k: (math.sqrt(L / m0) if regime.get(k) == 1.0
                 else math.sqrt(L) / max(abs(a0), 1e-12))
             for k, L in BASELINE_LOSSES.items() if k != "interface"}

    def loss_fn(n, c):
        y = n(c)
        out = {}
        for k, L in BASELINE_LOSSES.items():
            if k == "interface":
                out[k] = torch.zeros((), dtype=torch.float64)
                continue
            f = y[:, 0]
            if regime[k] == 1.0:
                out[k] = ((const[k] * f) ** 2).mean()
            else:
                # small value, O(1) gradient: r = eps + (f - <f>_0), where the
                # offset is a constant so grad r = grad f survives
                out[k] = (((f - f.detach()) + const[k] * f.mean()) ** 2).mean()
        for u, s in (("Mk", 3.861e-8), ("Mk_d", 1.874e-8), ("Tm", 1.863e-6)):
            out[f"pde_mech::{u}"] = out["pde_mech"] * (s / 1.624e-6)
        return out

    return net, loss_fn, colloc


def take_adam_steps(net, loss_fn, colloc, n, lr=1e-3, weights=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(n):
        L = losses_at(net, loss_fn, colloc)
        w = weights or {}
        total = sum(w.get(k, 1.0) * v for k, v in L.items() if "::" not in k)
        opt.zero_grad(); total.backward(); opt.step()

def snapshot(net, loss_fn, colloc):
    L = losses_at(net, loss_fn, colloc)
    params = [p for p in net.parameters() if p.requires_grad]
    return ({k: float(v.detach()) for k, v in L.items()},
            grad_norm_table(L, params))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="run against a synthetic loss with the 6cbd402 "
                         "magnitudes, to verify the plumbing")
    ap.add_argument("--adam-steps", type=int, default=0,
                    help="steps between the two snapshots; 0 skips the "
                         "regime fit (D-W.0b). 500 is the recommended value.")
    ap.add_argument("--anchor", default="bc_mech")
    ap.add_argument("--seed", type=int, default=20250812)
    ap.add_argument("--n", type=int, default=300,
                    help="collocation points per set. 300 matches the "
                         "baseline note; 3000 for a stable measurement.")
    ap.add_argument("--json", metavar="PATH",
                    help="write the raw numbers, for the methods section")
    a = ap.parse_args(argv)

    global build, losses_at
    if a.selftest:
        build = lambda s, n=None: _selftest_build(s)    # noqa: E731
        losses_at = lambda n, f, c: f(n, c)             # noqa: E731
        print("*** --selftest: synthetic loss, NOT the Isikdere physics ***\n")

    net, loss_fn, colloc = build(a.seed, a.n)
    L0, g0 = snapshot(net, loss_fn, colloc)

    missing = [k for k in TERMS if k not in L0]
    if missing:
        print(f"ERROR: losses_at() did not return {missing}", file=sys.stderr)
        return 2
    if all(v == 0.0 for v in g0.values()):
        print("ERROR: every gradient norm is zero. losses_at() is almost "
              "certainly detaching the terms.", file=sys.stderr)
        return 2

    p = None
    L1 = g1 = None
    if a.adam_steps:
        take_adam_steps(net, loss_fn, colloc, a.adam_steps)
        L1, g1 = snapshot(net, loss_fn, colloc)
        p = regime_exponents((L0, g0), (L1, g1))

    core = {k: v for k, v in g0.items() if "::" not in k}
    floor = BalancerConfig().activity_floor * max(core.values())
    print("Step 3.4 measurement 1 -- gradient norms at the Step 3.3 state")
    print(format_table(core, anchor=a.anchor, p=p))
    if p is None:
        print("\n(no p column: rerun with --adam-steps 500 for the regime "
              "fit, D-W.0b)")

    print("\nagainst the D-W.7 static seed:")
    ga = core[a.anchor]
    print(f"{'term':<16}{'w^ measured':>14}{'w0 seed':>14}{'c = w^/w0':>14}"
          f"  inside +/-3 orders?")
    for k in sorted(core, key=lambda t: -core[t]):
        w0 = PI_SEED.get(k, 1.0)
        if core[k] <= floor:
            print(f"{k:<16}{'inactive':>14}{w0:>14.3e}{'held':>14}"
                  f"  n/a (D-W.4)")
            continue
        c = (ga / core[k]) / w0
        print(f"{k:<16}{ga / core[k]:>14.3e}{w0:>14.3e}{c:>14.3e}"
              f"  {'yes' if -3 <= math.log10(c) <= 3 else 'NO -- reseed'}")

    sub = {k: v for k, v in g0.items() if "::" in k}
    if sub:
        print("\nper-stratum (D-W.5: reported, not weighted)")
        for k, v in sorted(sub.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<28}{v:>13.4e}")

    strays = [k for k, (b0, b1) in weight_bounds().items()
              if core.get(k, 0) > 0 and not b0 <= ga / core[k] <= b1]
    if strays:
        print(f"\nnote: {', '.join(strays)} measured outside the nominal "
              "loss-table bracket. Expected; see weight_bounds().")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"losses": L0, "grad_norms": g0,
                       "losses_1": L1, "grad_norms_1": g1, "p": p,
                       "anchor": a.anchor, "adam_steps": a.adam_steps,
                       "n": a.n, "seed": a.seed,
                       "selftest": a.selftest}, f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())