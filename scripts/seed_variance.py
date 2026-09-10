"""Is the measured `w^` column a measurement, or a sample of a heavy tail?

D-W.7 seeds the balancer from `w^ = g_anchor / g_i` measured once. That is only
legitimate if `w^` is a property of the OBJECTIVE rather than of the particular
collocation draw. This varies the sampling seed at fixed N, and N at fixed
sampling seed, and reports the spread per term.

Day 26 result (8x64, eps_psi = 0.3, t = 0):

    term            spread across 3 sampling seeds
                        N = 300        N = 1200
    pde_richards           2.0x         1264.1x
    pde_mech              45.0x           30.1x
    interface              9.1x            1.4x
    ic_disp                3.4x            3.1x
    bc                     2.5x            2.1x
    ic_head                2.4x            2.2x
    bc_mech                1.0x            1.0x   (anchor, 1 by definition)

The two PDE residual terms are not estimates of anything stable, and
`pde_richards` gets WORSE with more points -- the opposite of Monte-Carlo
convergence, and the signature of an estimator whose variance is dominated by
rare extreme samples. van Genuchten K_r spans nine orders across this domain
(1.7e-04 in Tm near the surface to 2.5e-13 at the crest), so the mean squared
Richards residual and its gradient are set by whichever few points land in the
extreme cells.

CONSEQUENCE: `PI_SEED_8X64`'s `pde_richards` entry was never a measurement.
The 3.466 in the 2 Sep file has an N=1200 counterpart of 0.0034 at another
sampling seed. Seeding from it is fitting a heavy tail, which is why the
16 Aug seed went stale, why the t=0 seed clips at +3, and why the geometric
mean of two such samples clips at -3. The analytic `pi_seed()` does not
depend on a draw and wins the three-arm ablation on five of seven terms.

The boundary and IC terms ARE stable (2-3x), so `ic_disp`'s 4.7e+04 is a real
requirement and not noise. That one needs the anchor decision, not a seed.

    PYTHONPATH=. python scripts/seed_variance.py --json docs/seed_variance_day26.json
"""
import argparse
import json
import sys

sys.path.insert(0, ".")

from scripts.grad_norm_table import build, losses_at          # noqa: E402
from src.weighting import grad_norm                           # noqa: E402


def w_hat(N, sampling_seed, anchor="bc_mech", layers=8, width=64, eps_psi=0.3):
    net, lf, coll = build(seed=sampling_seed, N=N, n_layers=layers,
                          n_neurons=width, eps_psi=eps_psi)
    params = list(net.parameters())
    g = {k: grad_norm(v, params)
         for k, v in losses_at(net, lf, coll).items()}
    ga = g[anchor]
    return {k: (ga / v if v > 0 else float("inf")) for k, v in g.items()}, g


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[300, 1200])
    ap.add_argument("--sampling-seeds", type=int, nargs="+",
                    default=[20250812, 7, 1234])
    ap.add_argument("--anchor", default="bc_mech")
    ap.add_argument("--eps-psi", type=float, default=0.3)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)

    out = {"anchor": a.anchor, "eps_psi": a.eps_psi, "layers": a.layers,
           "sampling_seeds": a.sampling_seeds, "runs": {}, "spread": {}}

    for N in a.n:
        acc, raw = {}, {}
        for sd in a.sampling_seeds:
            w, g = w_hat(N, sd, a.anchor, a.layers, eps_psi=a.eps_psi)
            raw[str(sd)] = {"w_hat": w, "grad_norms": g}
            for k, v in w.items():
                acc.setdefault(k, []).append(v)
        out["runs"][str(N)] = raw
        out["spread"][str(N)] = {k: max(v) / min(v) for k, v in acc.items()}

        print(f"\nN = {N}   (sampling seeds {a.sampling_seeds})")
        print(f"  {'term':<14}{'min':>12}{'max':>12}{'max/min':>11}  verdict")
        for k, v in sorted(acc.items(), key=lambda kv: -max(kv[1]) / min(kv[1])):
            s = max(v) / min(v)
            verdict = ("ANCHOR" if k == a.anchor else
                       "unusable" if s > 10 else
                       "noisy" if s > 5 else "stable")
            print(f"  {k:<14}{min(v):>12.4g}{max(v):>12.4g}{s:>11.1f}  {verdict}")

    print("\nA term whose w^ moves more than the +/-3 clip across sampling "
          "seeds\ncannot be seeded from a single measurement. See the module "
          "docstring.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
