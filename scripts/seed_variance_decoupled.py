r"""`seed_variance.py` varies the network AND the draw. Which one moves `w^`?

`grad_norm_table.build(seed=...)` passes ONE integer to two places:

    cfg  = dataclasses.replace(tiny(), ..., seed=seed)   -> PINN.__init__ calls
                                                           torch.manual_seed
    coll = sample_interior(N, seed)                      -> the collocation draw

So `scripts/seed_variance.py`, whose docstring says it "varies ONLY the
collocation draw", in fact varies the network initialisation at the same time.
The Day 26 headline -- `w^[pde_richards]` moving 1264x at N = 1200 -- is a
spread over BOTH. This script runs the 2x2: {network seed fixed, varied} x
{sampling seed fixed, varied}, everything else identical to `seed_variance.py`.

WHAT THE ARMS DO AND DO NOT TELL YOU. An earlier version of this docstring read
the draw arm as "the part better collocation can fix" and the init arm as "the
part it cannot". That was a hypothesis, it was tested, and it is WRONG. Running
this script with `--sampler front` takes the draw arm from 469.9x to 15989.5x:
concentrating collocation in the live band makes the spread nearly 34x worse.

The arms decompose the spread. They do not identify its cause. Both are
downstream of the same thing -- `scripts/residual_tail.py` shows 2-3 points out
of 4000 carrying 90% of `L_PDE_richards` under either sampler, all of them at
psi = -0.06 to -0.47 m, because `NearPhysical` is `psi* = psi0* + eps_psi*raw`
off a bare `nn.Linear` and at eps_psi = 0.3 between 15% and 35% of points
initialise at psi >= 0, saturated, in a domain that is unsaturated everywhere
by construction. Van Genuchten K_r with n = 1.2 is near-singular there. Which
arm moves more is a question about where the accident lands, not about what
the accident is.

    PYTHONPATH=. python scripts/seed_variance_decoupled.py --n 1200 \
        --json docs/seed_variance_decoupled_day27.json
"""
import argparse
import dataclasses
import json
import sys

sys.path.insert(0, ".")

from scripts.grad_norm_table import losses_at              # noqa: E402
from src.weighting import grad_norm                        # noqa: E402


def build2(net_seed, sampling_seed, N, n_layers=8, n_neurons=64,
           eps_psi=0.3, sampler="uniform"):
    """`grad_norm_table.build`, with the two seeds separated.

    Kept a near-copy rather than a refactor of `build` so the pinned artifact
    `docs/seed_variance_day26.json` and its test stay byte-reproducible.
    """
    from src.config import BOUNDS, tiny
    from src.loss import ic_targets
    from src.materials import load_materials
    from src.model import PINN, NearPhysical
    from src.sampling import (sample_boundary, sample_interfaces,
                              sample_interior)
    from src.sigma0 import attach_sigma0

    cfg = dataclasses.replace(tiny(), n_layers=n_layers,
                              n_neurons=n_neurons, seed=net_seed)
    net = NearPhysical(PINN(cfg, BOUNDS), eps_psi=eps_psi)

    if sampler == "front":
        from src.sampling_front import sample_interior_front
        coll = sample_interior_front(N, sampling_seed)
    else:
        coll = sample_interior(N, sampling_seed)
    attach_sigma0(coll)
    bcs = sample_boundary(N, sampling_seed, 30.0)
    for c in bcs.values():
        attach_sigma0(c)

    loss_fn = dict(bcs=bcs, ic=ic_targets(),
                   ifaces=sample_interfaces(N, sampling_seed),
                   mats=load_materials())
    return net, loss_fn, coll


def w_hat(net_seed, sampling_seed, N, anchor="bc_mech", **kw):
    net, lf, coll = build2(net_seed, sampling_seed, N, **kw)
    params = list(net.parameters())
    g = {k: grad_norm(v, params) for k, v in losses_at(net, lf, coll).items()}
    ga = g[anchor]
    return {k: (ga / v if v > 0 else float("inf")) for k, v in g.items()}


def spread(rows):
    acc = {}
    for r in rows:
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    return {k: max(v) / min(v) for k, v in acc.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20250812, 7, 1234])
    ap.add_argument("--anchor", default="bc_mech")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--eps-psi", type=float, default=0.3)
    ap.add_argument("--sampler", default="uniform",
                    choices=["uniform", "front"],
                    help="interior collocation sampler under test")
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    kw = dict(n_layers=a.layers, eps_psi=a.eps_psi, sampler=a.sampler)
    S = a.seeds
    fixed = S[0]

    arms = {
        # varies the draw only -- the experiment seed_variance.py claims to run
        "draw_only":  [(fixed, s) for s in S],
        # varies the initialisation only
        "init_only":  [(s, fixed) for s in S],
        # both together -- reproduces seed_variance.py
        "both":       [(s, s) for s in S],
    }

    out = {"N": a.n, "anchor": a.anchor, "seeds": S, "layers": a.layers,
           "eps_psi": a.eps_psi, "sampler": a.sampler, "arms": {}}

    results = {}
    for name, pairs in arms.items():
        rows = [w_hat(ns, ss, a.n, a.anchor, **kw) for ns, ss in pairs]
        results[name] = spread(rows)
        out["arms"][name] = {"pairs": pairs, "spread": results[name],
                             "w_hat": rows}

    terms = sorted(results["both"], key=lambda k: -results["both"][k])
    print(f"\nN = {a.n}, {a.layers}x64, eps_psi = {a.eps_psi}, "
          f"anchor = {a.anchor}, sampler = {a.sampler}, seeds {S}")
    print(f"  (net_seed, sampling_seed) pairs per arm:")
    for name, pairs in arms.items():
        print(f"    {name:<10} {pairs}")
    print(f"\n  {'term':<14}{'draw only':>12}{'init only':>12}"
          f"{'both':>12}{'d x i':>12}   larger arm")
    print("  " + "-" * 74)
    for k in terms:
        d, i, b = (results[arm][k] for arm in ("draw_only", "init_only", "both"))
        # 'larger arm' is descriptive only -- it says where the spread shows up,
        # NOT what causes it. See the module docstring.
        arm = ("anchor" if k == a.anchor else
               "draw" if d > 3 * i else
               "init" if i > 3 * d else "neither")
        out["arms"].setdefault("_compounding", {})[k] = d * i / b if b > 0 else None
        # .4g, not .1f: the front-sampler arm reaches 2.5e+09 and a fixed
        # decimal field silently overflows into the next column.
        print(f"  {k:<14}{d:>12.1f}{i:>12.1f}{b:>12.1f}{d * i:>12.4g}   {arm}")

    print("\n  'larger arm' is DESCRIPTIVE. It reports which factor the spread "
          "shows up under,\n  not which one causes it -- running this with "
          "--sampler front takes pde_richards'\n  draw arm from 469.9x to "
          "15989.5x. Both arms are downstream of the ansatz\n  reaching "
          "psi = 0; see the module docstring and scripts/residual_tail.py.")
    print("\n  'd x i' against 'both' tests whether the two sources are "
          "independent. Close\n  agreement means they compound multiplicatively "
          "and the decomposition is clean;\n  'both' far below the product "
          "means the same few extreme points drive each arm.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())