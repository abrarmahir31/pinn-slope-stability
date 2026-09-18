r"""Flux fraction per stratum -- the regime signal for the `activity_floor`
rewrite (Day 28 item 1).

    p = max(|diffusion|, |gravity|) / (|storage| + |diffusion| + |gravity|)

The definition lives in `src/residuals.flux_fraction`, which carries the
reasoning; this is the CLI around it. Computed POINTWISE and reduced with a
FRACTION rather than a quantile -- see the docstring there for why both of
those matter, and why no global threshold exists.

    PYTHONPATH=. python scripts/flux_fraction.py CKPT
    PYTHONPATH=. python scripts/flux_fraction.py --untrained          # smoke
    PYTHONPATH=. python scripts/flux_fraction.py CKPT --json docs/flux_fraction_day28.json
    PYTHONPATH=. python scripts/flux_fraction.py --why                # ceiling table

`--why` prints the per-material attainable ceiling from the D-S.4 sweep and
exits. It reads `docs/ds4_term_scales.json` if present and RECOMPUTES from
`src` otherwise, so it cannot go stale against the artifact silently.

Windows:  set PYTHONPATH=.   (not the inline `PYTHONPATH=. cmd` form)
"""
import argparse
import json

import torch

import paths
from src.config import bounds_from_geometry, full
from src.loss import psi_of
from src.materials import load_materials
from src.model import PINN, NearPhysical
from src.nondim import SCALES
from src.residuals import (attainable_flux_fraction, flux_fraction,
                           richards_residual)
from src.sampling import sample_interior

#: psi values of the D-S.4 sweep, mirrored from scripts/ds4_term_scales.py so
#: `--why` works without that artifact on disk.
PSI_M = (-0.01, -0.1, -1.0, -3.0, -10.0, -30.0, -80.0, -161.0)


def qs(v, ps=(1, 50, 99, 100)):
    return {f"p{p}": float(torch.quantile(v, p / 100.0)) for p in ps}


def ceiling_from_src(s=SCALES):
    """Attainable p per material. Thin wrapper over
    `residuals.attainable_flux_fraction` so the definition has one home and
    tests/test_flux_fraction.py recomputes the same thing this prints."""
    return {tag: attainable_flux_fraction(mat, s)
            for tag, mat in load_materials().items()}


def print_why():
    art = paths.DOCS / "ds4_term_scales.json"
    recomputed = ceiling_from_src()
    print("Attainable ceiling on p, over the D-S.4 psi sweep.")
    print("Network-independent: this is material physics, not training state.\n")
    print(f"{'unit':>6}{'ceiling (recomputed)':>24}{'from artifact':>16}")
    from_art = {}
    if art.exists():
        d = json.load(open(art))
        for r in d["rows"]:
            p = (max(r["D_over_S"], r["G_over_S"])
                 / (1 + r["D_over_S"] + r["G_over_S"]))
            from_art[r["unit"]] = max(from_art.get(r["unit"], 0.0), p)
    for tag in sorted(recomputed):
        a = from_art.get(tag)
        print(f"{tag:>6}{recomputed[tag]:>24.3e}"
              f"{(f'{a:.3e}' if a is not None else '-'):>16}")
    print("\nThe ceiling differs by 3.3 orders between Tm and the marls, and the\n"
          "marl ceiling is below any threshold anyone would write down. A single\n"
          "global cut on p is therefore an unconditional disable in Mk/Mk_d, not\n"
          "a regime rule. Normalise per material, or scope the rule to Tm and\n"
          "record that as a decision.")
    if from_art and any(
            abs(recomputed[t] - from_art[t]) > 1e-9 * max(1.0, from_art[t])
            for t in from_art):
        print("\nWARNING: recomputed and artifact disagree. Do not patch this\n"
              "table -- regenerate ds4_term_scales.json and reread D-S.4.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--untrained", action="store_true",
                    help="evaluate a fresh NearPhysical instead of a checkpoint")
    ap.add_argument("--mode", default=None,
                    choices=("unbounded", "cap", "exp"),
                    help="ansatz mode for --untrained. Default: NearPhysical's "
                         "own default. NOTE p is meaningless under 'unbounded' "
                         "(psi >= 0 points read p = 1.000).")
    ap.add_argument("--eps-psi", type=float, default=None,
                    help="ansatz prefactor the checkpoint was TRAINED with. "
                         "Anything before 2 Sep needs 3e-3.")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--sampling-seed", type=int, default=20260808)
    ap.add_argument("--t", type=float, nargs="+", default=[0.0, 1.0, 7.0, 30.0])
    ap.add_argument("--p-ref", type=float, default=1e-3,
                    help="reference for the reported live fraction (default "
                         "1e-3). NOT a calibrated threshold -- see --why.")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--why", action="store_true")
    a = ap.parse_args(argv)

    if a.why:
        print_why()
        return 0
    if not (a.ckpt or a.untrained):
        ap.error("give a checkpoint path, or --untrained for a smoke run")

    BOUNDS = bounds_from_geometry(t_max=30.0, s=SCALES)
    cfg = full()
    cfg.device, cfg.dtype = "cpu", "float64"
    step = None
    if a.ckpt:
        # Read the ansatz from the checkpoint. `mode` changes the FORWARD
        # PASS, so rebuilding with NearPhysical's default evaluates a cap/exp
        # run as unbounded and silently yields a different field from the
        # same weights. train.py stores it under cfg["ansatz"].
        d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        saved = d.get("cfg") or {}
        kw = {}
        eps = a.eps_psi if a.eps_psi is not None else saved.get("eps_psi")
        md = a.mode if a.mode is not None else saved.get("ansatz")
        ck = saved.get("cap_k")
        if eps is not None:
            kw["eps_psi"] = float(eps)
        if md is not None:
            kw["mode"] = md
        if ck is not None:
            kw["cap_k"] = float(ck)
        uv = saved.get("eps_uv")   # Day 42: rebuild at the
        if uv is not None:         # checkpoint's eps_uv, not the default
            kw["eps_uv"] = float(uv)
        if md is None:
            print("WARNING: checkpoint carries no 'ansatz' key; using "
                  "NearPhysical's default. If this was a cap/exp run the "
                  "field below is WRONG. Pass --mode explicitly.")
        net = NearPhysical(PINN(cfg, BOUNDS), **kw)
        net.load_state_dict(d["net"])
        step = d.get("step")
    else:
        kw = {} if a.eps_psi is None else {"eps_psi": a.eps_psi}
        if a.mode is not None:
            kw["mode"] = a.mode
        net = NearPhysical(PINN(cfg, BOUNDS), **kw)
    net.eval()

    print(f"checkpoint: {a.ckpt or '(untrained)'}   step: {step}")
    print(f"ansatz:     {net.summary()}")
    print(f"p_ref:      {a.p_ref:.3e}\n")

    coll = sample_interior(a.n, a.sampling_seed)
    mats = load_materials()
    tags = sorted(set(coll.tag.tolist()))
    # Resolved from the network, NOT from the CLI args: `a.mode` and
    # `a.eps_psi` are None whenever the flag was omitted, which would record
    # the artifact as mode=null. `NearPhysical`'s default is about to change
    # under D-A.1, so a null here becomes unrecoverable the moment it does.
    out = {"ckpt": a.ckpt, "step": step, "n": a.n, "p_ref": a.p_ref,
           "sampling_seed": a.sampling_seed,
           "eps_psi": net.eps_psi, "eps_uv": net.eps_uv,
           "mode": net.mode, "cap_k": net.cap_k,
           "ceiling": ceiling_from_src(), "rows": []}

    hdr = (f"{'t (d)':>6}{'tag':>6}{'n':>6}{'p p1':>11}{'p p50':>11}"
           f"{'p p99':>11}{'p max':>11}{'live':>8}{'psi>=0':>8}")
    print(hdr)
    print("-" * len(hdr))
    for t_days in a.t:
        for tag in tags:
            m = torch.as_tensor(coll.tag == tag)
            if not m.any():
                continue
            x = coll.x[m].reshape(-1, 1).clone().requires_grad_(True)
            z = coll.z[m].reshape(-1, 1).clone().requires_grad_(True)
            t = torch.full_like(x, t_days).requires_grad_(True)

            def fields(X, Z, T):
                return psi_of(net(X, Z, T))

            _, terms = richards_residual(fields, x, z, t, mats[tag],
                                         s=SCALES, return_terms=True)
            p = flux_fraction(terms)
            live = float((p > a.p_ref).double().mean())
            # The unphysical-saturation confound, reported alongside so a
            # high live fraction can never be read without it.
            psi = fields(x, z, t).detach().reshape(-1)
            sat = float((psi >= 0).double().mean())

            row = {"t_days": t_days, "tag": tag, "n": int(m.sum()),
                   "p": qs(p), "live_frac": live, "psi_ge0_frac": sat}
            out["rows"].append(row)
            q = row["p"]
            print(f"{t_days:>6.1f}{tag:>6}{int(m.sum()):>6}"
                  f"{q['p1']:>11.3e}{q['p50']:>11.3e}{q['p99']:>11.3e}"
                  f"{q['p100']:>11.3e}{live:>8.3f}{sat:>8.3f}")
        print()

    if any(r["psi_ge0_frac"] > 0 for r in out["rows"]):
        print("WARNING: psi >= 0 points present. p reads 1.000 on those by\n"
              "construction (saturated K_r is maximal), so the live fraction\n"
              "above is measuring the ansatz, not the physics. Do not\n"
              "calibrate a threshold from this run.\n")

    print("live = frac(p > p_ref). A FRACTION, not a quantile: bounded, so a\n"
          "handful of extreme points cannot drag it (1.04x across five draws\n"
          "against 2.54x for p99). p_ref is not calibrated -- the ceiling is a\n"
          "material property and differs by 3.3 orders; run --why.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())