#!/usr/bin/env python
"""SSR sweep driver — Steps 5.1 (MC) and 5.2 (GHB).

    python -m scripts.ssr_sweep --criterion MC \
        --baseline runs/prod/ckpt_final.pt --out runs/ssr_mc

**STATUS: WRITTEN, NOT RUN.** Everything below the `# --- adapter ---` line was
written against the day-28 snapshot without executing it: the container it was
written in has one CPU core, 3 GB of RAM, and no torch. The algebra it calls
(`src/strength.py`, `src/plasticity.py`, `src/failure_surface.py`) has 48
passing tests; this driver has none. Budget half a day to debug it against the
real `train.py` before trusting a number out of it, and run `--smoke` first.

**IT ALSO CANNOT WORK UNTIL THE YIELD TERM IS WIRED INTO `loss.py`.** The
day-28 `total_loss` is elastic; strength parameters do not enter it. See the
`src/plasticity.py` header. This script assumes `total_loss` has grown a
`yield_params=` argument and a `pde_yield` entry in its parts dict. It has not.
That wiring is the blocking task and it is yours -- it touches the balancer,
which needs a weight for the new term, and the L0 normalisation, which needs a
baseline value for it.

METHOD
------
Coarse sweep upward from `--srf-start` in `--srf-step` increments until the
failure signature appears, then bisect to `--bisect-tol`. At each SRF the
network is fine-tuned from the PREVIOUS converged SRF, not from the frozen
baseline -- warm-starting is standard SSR practice and it is what makes the
sweep affordable, but it does mean a bad step contaminates everything after it.
`--cold-start` reverts to fine-tuning from the baseline every time; use it to
check any FOS you intend to publish, at least once.

FAILURE SIGNATURE
-----------------
Three conditions, ALL required, checked in `_has_failed`. Requiring all three
is the point: any one of them alone fires on optimiser trouble as readily as on
slope failure, and an SSR sweep that mistakes the former for the latter returns
a confident wrong number rather than an error.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import train                                        # noqa: E402
from src import failure_surface as fsurf                         # noqa: E402
from src import plasticity as pl                                 # noqa: E402
from src import strength as st                                   # noqa: E402
from src.loss import total_loss                                  # noqa: E402
from src.materials import load_materials                         # noqa: E402
from src.mechanics import strain_star, total_stress_star, uv_of  # noqa: E402
from src.nondim import SCALES                                    # noqa: E402

MC_DESIGN = st.MCParams(c=4400.0, phi=math.radians(15.4))


# ---------------------------------------------------------------------------
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--criterion", choices=("MC", "GHB"), required=True)
    ap.add_argument("--baseline", required=True,
                    help="frozen converged checkpoint, e.g. ckpt_final.pt")
    ap.add_argument("--out", default="runs/ssr")

    g = ap.add_argument_group("sweep")
    g.add_argument("--srf-start", type=float, default=1.0)
    g.add_argument("--srf-step", type=float, default=0.05)
    g.add_argument("--srf-max", type=float, default=3.0)
    g.add_argument("--bisect-tol", type=float, default=0.01)
    g.add_argument("--cold-start", action="store_true")

    g = ap.add_argument_group("fine-tuning")
    g.add_argument("--epochs", type=int, default=1500)
    g.add_argument("--lr", type=float, default=3e-4,
                   help="well below the baseline LR: this is a perturbation, "
                        "not a retrain")
    g.add_argument("--w-yield", type=float, default=1.0,
                   help="yield-penalty weight. THE FOS DEPENDS ON THIS -- "
                        "sweep it (0.3/1/3) and report the plateau")

    g = ap.add_argument_group("failure detection")
    g.add_argument("--plateau-factor", type=float, default=20.0,
                   help="total loss this many times the SRF=1 value")
    g.add_argument("--disp-factor", type=float, default=10.0)
    g.add_argument("--admissible-floor", type=float, default=0.80)

    g = ap.add_argument_group("GHB")
    g.add_argument("--sig3-lo", type=float, default=0.0)
    g.add_argument("--sig3-hi", type=float, default=8.5e5,
                   help="Pa. Phase-1 design_sig3max is 850 kPa = 0.25*gamma*H. "
                        "SET THIS FROM THE BASELINE STRESS FIELD along the "
                        "failure zone and report it with the FOS")

    ap.add_argument("--smoke", action="store_true",
                    help="3 SRF steps, 50 epochs each, no bisection")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
def _sig3_range(a):
    return (a.sig3_lo, a.sig3_hi)


def _yield_params(a, mats, srf):
    """Reduced strength parameters, per material tag."""
    out = {}
    for tag, mat in mats.items():
        out[tag] = pl.reduced_material_params(
            mat, a.criterion, srf,
            mc_base=MC_DESIGN if a.criterion == "MC" else None,
            sig3_range=_sig3_range(a) if a.criterion == "GHB" else None)
    return out


@torch.no_grad()
def _disp_norm(net, coll):
    u, v = uv_of(net(coll.x, coll.z, coll.t))
    return float(torch.sqrt(torch.mean(u ** 2 + v ** 2)).item())


def _diagnostics(net, coll, mats, yparams, criterion):
    """Admissible fraction and localisation, on the collocation set."""
    frac, worst = [], 0.0
    for tag, mat in mats.items():
        idx = np.flatnonzero(coll.tag == tag)
        if idx.size == 0:
            continue
        x = coll.x[idx].detach().clone().requires_grad_(True)
        z = coll.z[idx].detach().clone().requires_grad_(True)
        t = coll.t[idx].detach().clone().requires_grad_(True)

        (sxx, szz, sxz), _, _ = total_stress_star(net, x, z, t, mat, s=SCALES)
        sig1, sig3 = st.principal_stresses_from_cartesian(
            sxx.detach(), szz.detach(), sxz.detach())
        sig1, sig3 = sig1 * SCALES.sig_ref, sig3 * SCALES.sig_ref

        frac.append(pl.admissible_fraction(sig1, sig3, yparams[tag],
                                           criterion,
                                           sigma_scale=SCALES.sig_ref))
        worst = max(worst, float(torch.max(
            pl.yield_violation(sig1, sig3, yparams[tag], criterion,
                               sigma_scale=SCALES.sig_ref)).item()))
    return (float(np.mean(frac)) if frac else float("nan")), worst


def _has_failed(rec, ref, a):
    """All three conditions, deliberately.

    A plateaued loss alone is an optimiser that got stuck. A displacement jump
    alone is a network drifting in a null direction the BCs do not pin. A drop
    in admissible fraction alone can be one bad collocation point. Together
    they are a slope that will not stand up.
    """
    return (rec["total"] > a.plateau_factor * ref["total"]
            and rec["disp"] > a.disp_factor * ref["disp"]
            and rec["admissible"] < a.admissible_floor)


# --- adapter ---------------------------------------------------------------
# Everything below assumes loss.total_loss accepts `yield_params` and
# `w_yield`. It does not yet. This is the wiring task.
def _finetune(net, coll, loss_fn, a, yparams, epochs):
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    last = None
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        L, parts = total_loss(net, coll, loss_fn["bcs"], loss_fn["ic"],
                              loss_fn["ifaces"], loss_fn["mats"],
                              s=SCALES, include_mechanics=True, per_term=True,
                              yield_params=yparams, w_yield=a.w_yield)
        L.backward()
        opt.step()
        last = (float(L.item()), {k: float(v) for k, v in parts.items()})
    return last


def run(a):
    os.makedirs(a.out, exist_ok=True)
    logp = os.path.join(a.out, "sweep.jsonl")

    done = {}
    if os.path.exists(logp):
        for line in open(logp):
            r = json.loads(line)
            done[round(r["srf"], 6)] = r
        print(f"resuming: {len(done)} SRF points already on disk")

    targs = train.parse([])
    net, loss_fn, coll = train.setup(targs)
    ckpt = torch.load(a.baseline, map_location=net.device)
    net.load_state_dict(ckpt["net"])
    baseline_state = {k: v.clone() for k, v in net.state_dict().items()}
    mats = load_materials()

    epochs = 50 if a.smoke else a.epochs
    log = open(logp, "a")

    def evaluate(srf):
        key = round(srf, 6)
        if key in done:
            return done[key]
        if a.cold_start:
            net.load_state_dict(baseline_state)
        yp = _yield_params(a, mats, srf)
        t0 = time.time()
        total, parts = _finetune(net, coll, loss_fn, a, yp, epochs)
        adm, worst = _diagnostics(net, coll, mats, yp, a.criterion)
        rec = {"srf": srf, "total": total, "parts": parts,
               "disp": _disp_norm(net, coll), "admissible": adm,
               "worst_violation": worst, "sec": time.time() - t0,
               "criterion": a.criterion, "w_yield": a.w_yield,
               "cold_start": bool(a.cold_start),
               "sig3_range": _sig3_range(a) if a.criterion == "GHB" else None}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        done[key] = rec
        print(f"  SRF {srf:6.3f}  L={total:.4e}  adm={adm:.3f}  "
              f"|u|={rec['disp']:.3e}  {rec['sec']:.0f}s")
        return rec

    print(f"\n=== {a.criterion} SSR sweep ===")
    ref = evaluate(a.srf_start)

    stable, failed = a.srf_start, None
    srf = a.srf_start + a.srf_step
    n_max = 3 if a.smoke else int((a.srf_max - a.srf_start) / a.srf_step) + 1

    for _ in range(n_max):
        if srf > a.srf_max:
            break
        rec = evaluate(srf)
        if _has_failed(rec, ref, a):
            failed = srf
            break
        stable = srf
        srf += a.srf_step

    if failed is None:
        print(f"\nNO FAILURE up to SRF {a.srf_max}. The slope is either very "
              f"stable or the yield term is not binding -- check that "
              f"`admissible` fell at all. If it stayed at 1.000 the whole way, "
              f"strength never entered the loss and the sweep is measuring "
              f"nothing.")
        return

    if a.smoke:
        print(f"\nsmoke run: bracketed [{stable}, {failed}], stopping")
        return

    print(f"\nbracketed: stable {stable:.3f}, failed {failed:.3f} -- bisecting")
    while failed - stable > a.bisect_tol:
        mid = 0.5 * (stable + failed)
        if _has_failed(evaluate(mid), ref, a):
            failed = mid
        else:
            stable = mid

    fos = 0.5 * (stable + failed)
    print(f"\nFOS_{a.criterion} = {fos:.3f}  +/- {0.5*(failed-stable):.3f}")
    print("This number is not reportable until you have repeated it at "
          "w_yield 0.3 and 3.0 and shown the plateau.")

    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump({"criterion": a.criterion, "fos": fos,
                   "bracket": [stable, failed], "w_yield": a.w_yield,
                   "cold_start": bool(a.cold_start), "epochs": a.epochs,
                   "sig3_range": _sig3_range(a) if a.criterion == "GHB" else None,
                   "baseline": a.baseline}, f, indent=2)


if __name__ == "__main__":
    run(parse())