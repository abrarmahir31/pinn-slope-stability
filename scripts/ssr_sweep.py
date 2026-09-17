#!/usr/bin/env python
"""SSR sweep driver -- Steps 5.1 (MC) and 5.2 (GHB). Rewritten Day 40.

    python scripts/ssr_sweep.py --criterion GHB --baseline CKPT --out runs/ssr_ghb_w1 \
        --w-yield 1 --yield-norm ref --admissible-metric area \
        --admissible-drop 0.10 --sig3-lo 0 --sig3-hi 8.5e5

WHAT CHANGED FROM THE DAY-39 VERSION, and why each matters (none of it had
ever executed):

 1. The network is rebuilt from the CHECKPOINT's cfg (layers, width, ansatz,
    eps_psi, cap_k, sampling sizes). The old `train.setup(train.parse([]))`
    built a 2-layer CPU net at N_PDE = 500 and could not load an 8x64
    baseline; had it loaded, it would have broken D-A.3.
 2. `--criterion` reaches the loss. The old call never passed
    `yield_criterion`, so an MC sweep trained against the GHB penalty.
 3. The admissible fraction is read from `loss.L_yield` itself, so it is
    computed on EXACTLY the stress the penalty sees (sigma_0 + increment).
    The old `_diagnostics` omitted sigma_0, tested the network increment
    alone (eps_uv = 1e-3), and would have held admissibility near 1 so that
    `_has_failed` could never fire.
 4. Kozeny-Carman feedback follows the checkpoint (`no_feedback`) unless an
    arm overrides it. The old call defaulted to feedback OFF, so every FOS
    would have been one-way.
 5. The objective is the one the baseline minimised: the balancer's frozen
    weights over the seven `losses_at` terms, plus the yield term. The old
    call used `total_loss` with unit weights, a different function, so the
    field moved at SRF = 1 before any strength was reduced.
 6. Modelling choices have no defaults: `--w-yield`, `--yield-norm`,
    `--admissible-metric`, `--admissible-drop`, and for GHB `--sig3-lo/hi`.
    The old defaults (floor 0.80 absolute, sig3 0-850 kPa) silently defeated
    `reduced_material_params`'s refusal, and an absolute floor fires at
    SRF = 1 because the Day 40 recount puts ~45% of Mk_d outside GHB there.

Also new: a checkpoint per evaluated SRF (Fig 9 needs the critical states),
resume that restores the warm-start chain, non-finite losses reported as a
training failure rather than counted as slope failure, `--overrides` for the
Phase 6 arms, and every threshold written to result.json.

STILL DECISIONS, recorded but not settled here: soft constraint vs return
mapping (this is the soft arm); MC applies the bedding residual pair
c = 4.4 kPa, phi = 15.4 deg to EVERY stratum including Tm; the yield check is
on total stress with the Bishop increment, as loss.L_yield docstring states;
`--plateau-factor` and `--disp-factor` keep their Day-39 defaults.

FAILURE SIGNATURE: all three of loss plateau, displacement jump, and a DROP in
the chosen admissible metric relative to the SRF-start reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts import train                                        # noqa: E402
from src import arms                                             # noqa: E402
from src import plasticity as pl                                 # noqa: E402
from src import strength as st                                   # noqa: E402
from src.loss import L_BC, L_BC_mech, L_IC, L_PDE, L_PDE_mech, \
    L_interface, L_yield                                         # noqa: E402
from src.mechanics import uv_of                                  # noqa: E402
from src.nondim import SCALES                                    # noqa: E402
from src.weighting import TERMS                                  # noqa: E402

MC_DESIGN = st.MCParams(c=4400.0, phi=math.radians(15.4))
_CFG_KEYS = ("layers", "width", "seed", "eps_psi", "ansatz", "cap_k",
             "n_pde", "n_bc", "n_iface", "t_max")


# ---------------------------------------------------------------------------
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--criterion", choices=("MC", "GHB"), required=True)
    ap.add_argument("--baseline", required=True, help="checkpoint to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None,
                    help="default: cuda if available, else cpu")
    ap.add_argument("--overrides", default=None,
                    help="arm JSON from scripts/make_arms.py (Phase 6)")

    g = ap.add_argument_group("sampling (default: from the checkpoint)")
    g.add_argument("--n-pde", type=int, default=None)
    g.add_argument("--n-bc", type=int, default=None)
    g.add_argument("--n-iface", type=int, default=None)
    g.add_argument("--sampling-seed", type=int, default=None,
                   help="default: the checkpoint's seed. Change for the "
                        "reproducibility arm only.")

    g = ap.add_argument_group("sweep")
    g.add_argument("--srf-start", type=float, default=1.0)
    g.add_argument("--srf-step", type=float, default=0.05)
    g.add_argument("--srf-max", type=float, default=3.0)
    g.add_argument("--bisect-tol", type=float, default=0.01)
    g.add_argument("--cold-start", action="store_true",
                   help="fine-tune every SRF from the baseline, not the "
                        "previous SRF. Run once for any FOS you report.")

    g = ap.add_argument_group("fine-tuning -- DECISIONS, no defaults")
    g.add_argument("--epochs", type=int, default=1500)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--w-yield", type=float, required=True)
    g.add_argument("--yield-norm", choices=("raw", "ref"), required=True,
                   help="raw: w_yield * L_yield. ref: w_yield * L_yield / "
                        "L_yield(baseline net, SRF start), so w_yield is "
                        "relative to the yield term's own starting size.")

    g = ap.add_argument_group("failure detection")
    g.add_argument("--plateau-factor", type=float, default=20.0)
    g.add_argument("--disp-factor", type=float, default=10.0)
    g.add_argument("--admissible-metric", required=True,
                   help="area | min_tag | tag:<Mk|Mk_d|Tm>")
    g.add_argument("--admissible-drop", type=float, required=True,
                   help="failure needs the metric to FALL by more than this "
                        "(fraction, e.g. 0.10) below its SRF-start value")

    g = ap.add_argument_group("GHB -- required for --criterion GHB")
    g.add_argument("--sig3-lo", type=float, default=None, help="Pa")
    g.add_argument("--sig3-hi", type=float, default=None, help="Pa")

    ap.add_argument("--smoke", action="store_true",
                    help="3 SRF steps, 50 epochs each, no bisection")
    a = ap.parse_args(argv)

    if a.criterion == "GHB":
        if a.sig3_lo is None or a.sig3_hi is None:
            ap.error("--criterion GHB needs --sig3-lo and --sig3-hi. The fit "
                     "range is a modelling choice; see scripts/sig3_range.py")
        if not a.sig3_hi > a.sig3_lo:
            ap.error("--sig3-hi must exceed --sig3-lo")
    elif a.sig3_lo is not None or a.sig3_hi is not None:
        ap.error("--sig3-lo/--sig3-hi only apply to --criterion GHB")
    try:
        _parse_metric(a.admissible_metric)
    except ValueError as e:
        ap.error(str(e))
    if not 0.0 < a.admissible_drop < 1.0:
        ap.error("--admissible-drop must be in (0, 1)")
    if not a.w_yield > 0:
        ap.error("--w-yield must be > 0")
    return a


# ---------------------------------------------------------------------------
def _parse_metric(spec: str):
    if spec in ("area", "min_tag"):
        return spec, None
    if spec.startswith("tag:") and spec[4:] in arms.SECTION5_TAGS:
        return "tag", spec[4:]
    raise ValueError(f"--admissible-metric {spec!r}: use area, min_tag or "
                     f"tag:<{'|'.join(arms.SECTION5_TAGS)}>")


def _sig3_range(a):
    if a.criterion != "GHB":
        return None
    if a.sig3_lo is None or a.sig3_hi is None:
        raise ValueError("GHB needs an explicit sig3 range")
    return (float(a.sig3_lo), float(a.sig3_hi))


def _yield_params(a, mats, srf):
    """Reduced strength parameters per tag. MC uses the design pair for every
    tag (see module docstring); GHB reduces each material's own set."""
    return {tag: pl.reduced_material_params(
                mat, a.criterion, srf,
                mc_base=MC_DESIGN if a.criterion == "MC" else None,
                sig3_range=_sig3_range(a))
            for tag, mat in mats.items()}


def _area_shares(coll) -> dict:
    w = coll.w.detach().cpu().numpy().ravel()
    return {t: float(w[coll.tag == t].sum() / w.sum())
            for t in sorted(set(coll.tag.tolist()))}


def _admissible_metric(adm: dict, shares: dict, spec: str) -> float:
    kind, tag = _parse_metric(spec)
    if kind == "area":
        return float(sum(shares[t] * adm[t] for t in adm))
    if kind == "min_tag":
        return float(min(adm.values()))
    if tag not in adm:
        raise KeyError(f"no admissible fraction for {tag}; have {sorted(adm)}")
    return float(adm[tag])


def _diagnostics(net, coll, mats, yparams, criterion, spec, s=SCALES):
    """Admissible fractions from `loss.L_yield` -- the penalty's own stress."""
    yld, parts = L_yield(net, coll, mats, yparams, criterion=criterion,
                         s=s, per_tag=True)
    adm = {k[len("admissible_"):]: float(v) for k, v in parts.items()
           if k.startswith("admissible_")}
    shares = _area_shares(coll)
    return {"admissible": _admissible_metric(adm, shares, spec),
            "admissible_by_tag": adm, "area_shares": shares,
            "pde_yield": float(yld.detach()),
            "pde_yield_by_tag": {k[len("pde_yield_"):]: float(v)
                                 for k, v in parts.items()
                                 if k.startswith("pde_yield_")}}


def _has_failed(rec, ref, a) -> bool:
    """All three, deliberately. Each alone fires on optimiser trouble."""
    return bool(rec["total"] > a.plateau_factor * ref["total"]
                and rec["disp"] > a.disp_factor * ref["disp"]
                and (ref["admissible"] - rec["admissible"]) > a.admissible_drop)


# ---------------------------------------------------------------------------
def _losses(net, loss_fn, coll, kc):
    """`grad_norm_table.losses_at` with the KC config passed through, so the
    'KC everywhere' arm is reachable. test_ssr_sweep binds the two."""
    b, mats = loss_fn, loss_fn["mats"]
    out = {"pde_richards": L_PDE(net, coll, mats, kc=kc)[0],
           "pde_mech": L_PDE_mech(net, coll, mats)[0],
           "bc": L_BC(net, b["bcs"], mats)[0],
           "bc_mech": L_BC_mech(net, b["bcs"], mats)[0],
           "interface": L_interface(net, b["ifaces"], mats)[0]}
    _, ic = L_IC(net, *b["ic"], keep_graph=True)
    out["ic_head"], out["ic_disp"] = ic["ic_head"], ic["ic_disp"]
    return out


def _frozen_total(L: dict, w: dict):
    """sum_k w_k L_k over TERMS, as LossBalancer.total. Missing terms raise."""
    missing = [k for k in TERMS if k not in L or k not in w]
    if missing:
        raise KeyError(f"objective missing {missing}")
    out = None
    for k in TERMS:
        out = w[k] * L[k] if out is None else out + w[k] * L[k]
    return out


def _train_args(ckpt_cfg: dict, a):
    """train.parse namespace reproducing the baseline's network and sampling."""
    missing = [k for k in ("layers", "width", "ansatz", "eps_psi")
               if k not in ckpt_cfg]
    if missing:
        raise KeyError(f"checkpoint cfg lacks {missing}; cannot rebuild the "
                       f"network it was trained with (D-A.3)")
    t = train.parse([])
    for k in _CFG_KEYS:
        if k in ckpt_cfg and ckpt_cfg[k] is not None:
            setattr(t, k, ckpt_cfg[k])
    for k in ("n_pde", "n_bc", "n_iface"):
        if getattr(a, k) is not None:
            setattr(t, k, getattr(a, k))
    if a.sampling_seed is not None:
        t.seed = a.sampling_seed
    t.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return t


def _kc_name(ckpt_cfg: dict, overrides: dict) -> str:
    if "kc" in overrides:
        return overrides["kc"]
    return "off" if ckpt_cfg.get("no_feedback", False) else "default"


@torch.no_grad()
def _disp_norm(net, coll):
    u, v = uv_of(net(coll.x, coll.z, coll.t))
    return float(torch.sqrt(torch.mean(u ** 2 + v ** 2)).item())


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
def run(a):
    os.makedirs(a.out, exist_ok=True)
    logp = os.path.join(a.out, "sweep.jsonl")
    state_dir = os.path.join(a.out, "states")
    os.makedirs(state_dir, exist_ok=True)

    ckpt = torch.load(a.baseline, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg") or {}
    overrides = arms.load_arm_file(a.overrides)["overrides"] if a.overrides else {}
    targs = _train_args(cfg, a)
    net, loss_fn, coll = train.setup(targs)
    net.load_state_dict(ckpt["net"])
    baseline_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    w_frozen = dict(ckpt["bal"]["w"])
    kc_name = _kc_name(cfg, overrides)
    kc = arms.kc_config(kc_name)
    loss_fn["mats"] = arms.apply_material_overrides(loss_fn["mats"], overrides)
    mats = loss_fn["mats"]
    epochs = 50 if a.smoke else a.epochs
    print(f"{net.summary()}\nN_PDE={targs.n_pde} N_BC={targs.n_bc} "
          f"device={net.device} KC={kc_name} overrides={overrides or 'none'}")
    if not arms.sigma0_consistent(overrides):
        print("WARNING: this arm moves an input of the offline sigma_0 solve; "
              "sigma_0 is NOT re-equilibrated for it.")

    with arms.boundary_overrides(mats, overrides):
        # Yield normalisation, measured once on the unmodified baseline net.
        yp0 = _yield_params(a, mats, a.srf_start)
        y0 = float(L_yield(net, coll, mats, yp0, criterion=a.criterion)[0].detach())
        if a.yield_norm == "ref":
            if not y0 > 0:
                raise RuntimeError(
                    f"--yield-norm ref: L_yield at SRF {a.srf_start} on the "
                    f"baseline is {y0:g}; nothing to normalise by. Use raw.")
            y_scale = y0
        else:
            y_scale = 1.0
        print(f"L_yield(baseline, SRF {a.srf_start}) = {y0:.4e}   "
              f"yield scale = {y_scale:.4e}")

        done, last_key = {}, None
        if os.path.exists(logp):
            for line in open(logp):
                r = json.loads(line)
                done[round(r["srf"], 6)] = r
                last_key = round(r["srf"], 6)
            if last_key is not None and not a.cold_start:
                sp = os.path.join(state_dir, f"srf_{last_key:.6f}.pt")
                if not os.path.exists(sp):
                    raise FileNotFoundError(
                        f"resume: {sp} missing, so the warm-start chain cannot "
                        f"be restored. Delete {logp} to restart.")
                net.load_state_dict(torch.load(sp, map_location=net.device,
                                               weights_only=False)["net"])
            print(f"resuming: {len(done)} SRF points on disk")

        def evaluate(srf):
            key = round(srf, 6)
            if key in done:
                return done[key]
            if a.cold_start:
                net.load_state_dict(baseline_state)
            yp = _yield_params(a, mats, srf)
            opt = torch.optim.Adam(net.parameters(), lr=a.lr)
            t0 = time.time()
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                L = _losses(net, loss_fn, coll, kc)
                phys = _frozen_total(L, w_frozen)
                yld = L_yield(net, coll, mats, yp, criterion=a.criterion)[0]
                total = phys + a.w_yield * yld / y_scale
                total.backward()
                opt.step()
            finite = bool(torch.isfinite(total).item())
            rec = {"srf": srf, "total": float(total.detach()),
                   "physics": float(phys.detach()),
                   "yield_term": float(yld.detach()),
                   "terms": {k: float(v.detach()) for k, v in L.items()},
                   "disp": _disp_norm(net, coll), "finite": finite,
                   "sec": time.time() - t0}
            if finite:
                rec.update(_diagnostics(net, coll, mats, yp, a.criterion,
                                        a.admissible_metric))
            torch.save({"net": net.state_dict(), "srf": srf, "cfg": cfg,
                        "sweep": vars(a)},
                       os.path.join(state_dir, f"srf_{key:.6f}.pt"))
            with open(logp, "a") as log:
                log.write(json.dumps(rec) + "\n")
            done[key] = rec
            adm = rec.get("admissible", float("nan"))
            by = "  ".join(f"{t}={v:.3f}" for t, v in
                           sorted(rec.get("admissible_by_tag", {}).items()))
            print(f"  SRF {srf:6.3f}  L={rec['total']:.4e}  adm={adm:.3f} "
                  f"[{by}]  |u|={rec['disp']:.3e}  {rec['sec']:.0f}s")
            if not finite:
                raise FloatingPointError(
                    f"non-finite loss at SRF {srf}: a TRAINING failure, not a "
                    f"slope failure. Not counted toward FOS.")
            return rec

        print(f"\n=== {a.criterion} SSR sweep, w_yield={a.w_yield} "
              f"({a.yield_norm}) ===")
        ref = evaluate(a.srf_start)
        stable, failed, srf = a.srf_start, None, a.srf_start
        n_max = 3 if a.smoke else int(round((a.srf_max - a.srf_start)
                                            / a.srf_step))
        for i in range(1, n_max + 1):
            srf = round(a.srf_start + i * a.srf_step, 6)
            if srf > a.srf_max + 1e-9:
                break
            rec = evaluate(srf)
            if _has_failed(rec, ref, a):
                failed = srf
                break
            stable = srf

        result = {"criterion": a.criterion, "fos": None,
                  "bracket": [stable, failed], "w_yield": a.w_yield,
                  "yield_norm": a.yield_norm, "yield_scale": y_scale,
                  "L_yield_baseline": y0,
                  "admissible_metric": a.admissible_metric,
                  "admissible_drop": a.admissible_drop,
                  "admissible_ref": ref["admissible"],
                  "admissible_ref_by_tag": ref.get("admissible_by_tag"),
                  "plateau_factor": a.plateau_factor,
                  "disp_factor": a.disp_factor, "cold_start": a.cold_start,
                  "epochs": epochs, "lr": a.lr, "sig3_range": _sig3_range(a),
                  "mc_design": ([MC_DESIGN.c, math.degrees(MC_DESIGN.phi)]
                                if a.criterion == "MC" else None),
                  "kc": kc_name, "overrides": overrides,
                  "arm": (arms.load_arm_file(a.overrides) if a.overrides else None),
                  "sigma0_consistent": arms.sigma0_consistent(overrides),
                  "baseline": a.baseline, "n_pde": targs.n_pde,
                  "sampling_seed": targs.seed, "smoke": a.smoke,
                  "git_commit": _git_commit()}

        if failed is None:
            print(f"\nNO FAILURE up to SRF {srf:.3f}. Check the admissible "
                  f"column: if it never fell, strength never bound.")
            result["status"] = "no_failure"
        elif a.smoke:
            print(f"\nsmoke run: bracketed [{stable}, {failed}], stopping")
            result["status"] = "smoke_bracketed"
        else:
            print(f"\nbracketed: stable {stable:.3f}, failed {failed:.3f}")
            while failed - stable > a.bisect_tol:
                mid = round(0.5 * (stable + failed), 6)
                if _has_failed(evaluate(mid), ref, a):
                    failed = mid
                else:
                    stable = mid
            result.update(fos=0.5 * (stable + failed), bracket=[stable, failed],
                          status="fos")
            print(f"\nFOS_{a.criterion} = {result['fos']:.3f} +/- "
                  f"{0.5 * (failed - stable):.3f}   (w_yield {a.w_yield}; "
                  f"not reportable without the w_yield plateau)")

    with open(os.path.join(a.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {os.path.join(a.out, 'result.json')}")
    return result


if __name__ == "__main__":
    run(parse())
