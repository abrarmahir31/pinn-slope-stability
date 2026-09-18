"""
train.py -- Step 4.1 training loop with the Step 3.4 adaptive weighting.

    # toy, reproduces docs/step34_toyrun.json
    PYTHONPATH=. python scripts/train.py --epochs 1000 --n-pde 500 --n-bc 500

    # lab PC smoke test
    PYTHONPATH=. python scripts/train.py --layers 8 --n-pde 10000 --n-bc 3000 \
        --epochs 2000 --device cuda --ckpt-every 500 --out runs/smoke

    # resume
    PYTHONPATH=. python scripts/train.py --resume runs/smoke/ckpt_001000.pt \
        --epochs 4000 --out runs/smoke

    # Adam then L-BFGS polish (Step 4.1b). Weights freeze at the handover;
    # see _lbfgs_phase. ckpt_adam_final.pt is written there regardless of
    # --ckpt-every, so the pre-polish field stays recoverable.
    PYTHONPATH=. python scripts/train.py --layers 8 --n-pde 10000 \
        --epochs 2000 --lbfgs-epochs 200 --device cuda --out runs/polish

HOW TO READ THE LOG -- the one thing that will mislead you
-----------------------------------------------------------
NOT the weighted total. It STEPS at every weight update. That is arithmetic,
not divergence: sum_i w_i L_i changes discontinuously when w changes, and a
run that is working perfectly shows a sawtooth. Anyone watching `total` for a
smooth decrease will conclude the run is broken when it is not.

Read the per-term UNWEIGHTED losses. Those are printed as ratios to their own
step-0 value (the columns) and stored raw in log.jsonl under "L".

Records carry "phase": "adam" or "lbfgs". L-BFGS rows are prefixed `L ` in the
console. In the L-BFGS phase "spread" and "grad_norms" are NULL rather than
repeated -- the balancer is frozen there, so echoing its last measurement would
draw a flat line that looks measured and is a stale cache.

A FOURTH SIGNAL, added Day 27, and it is the one that would have caught the
failure the other three missed: "psi_sat_frac" and "psi_max_m". Section 5 is
unsaturated everywhere (Z_WT = 197 m sits below Z_BASE = 200 m), so psi >= 0 is
unphysical, and a psi_max of -20 m or drier under a 30-day rain BC means the
field has left the live band and the Richards residual is being satisfied
structurally rather than solved. A 2000-epoch run can pass all three signals
above while doing exactly that.

Three health signals, all in log.jsonl:
  * each L_i trending down. In the 16 Aug toy run at 2x64 every term
    improved: pde_mech 6.0e-05, pde_richards 9.3e-04, bc 4.3e-04,
    ic_head 9.7e-04, ic_disp 0.055, bc_mech 0.973 of their starts.
  * "spread" -> 1. It reached 27.6 in that run, bounded below by `bc`
    sitting on the c = 1e3 clip. Rising spread means the balancer is
    losing ground.
  * "clipped" short. A term at the clip has a weight no longer set by
    measurement. `bc` clipping is known and open; anything else is new.

WHAT COSTS TIME AND MEMORY
--------------------------
Peak VRAM is DURING bal.update(), not in steady state: seven autograd.grad
calls with retain_graph=True before the training backward. It will not show
in nvidia-smi between updates. If it OOMs, raise --every before cutting
--n-pde; the balancer is far cheaper at every=200 and barely less accurate.

L_BC_mech clones fresh leaf tensors per material tag per segment -- three
tags x six segments, every forward pass. At n_bc in the thousands that is the
likely bottleneck, not the interior residual. Profile it before assuming
N_PDE is the problem.

Author: <thesis>  |  Phase 4, Step 4.1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grad_norm_table import losses_at          # the seven-term assembly

from src.config import BOUNDS, tiny
from src.loss import ic_targets
from src.materials import load_materials
from src.model import PINN, NearPhysical
from src.sampling import (sample_boundary, sample_interfaces, sample_interior,
                          to_device)
from src.nondim import SCALES
from src.sigma0 import attach_sigma0
from src.weighting import (PI_SEED_8X64, PI_SEED_8X64_T0,
                           BalancerConfig, LossBalancer, TERMS)
from src.weighting import pi_seed
def _atomic_save(payload, path):
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

# ---------------------------------------------------------------------------
def setup(a):
    """(net, loss_fn, coll), all on a.device.

    sigma_0 is attached BEFORE the move: attach_sigma0 does a numpy-side
    lookup against the cache and expects CPU tensors. to_device then rebuilds
    x/z/t as leaves -- a plain .to() on a leaf with requires_grad returns a
    NON-leaf, and autograd.grad(psi, x) silently resolves to the wrong node.
    """
    cfg = dataclasses.replace(tiny(), n_layers=a.layers, n_neurons=a.width,
                              seed=a.seed, device=a.device)
    uv = {} if getattr(a, "eps_uv", None) is None else {"eps_uv": a.eps_uv}
    net = NearPhysical(PINN(cfg, BOUNDS), eps_psi=a.eps_psi,
                       mode=a.ansatz, cap_k=a.cap_k, **uv)
    dev = net.device

    coll = sample_interior(a.n_pde, a.seed)
    attach_sigma0(coll)
    coll = to_device(coll, dev)

    bcs = {}
    for seg, c in sample_boundary(a.n_bc, a.seed, a.t_max).items():
        attach_sigma0(c)
        bcs[seg] = to_device(c, dev)

    ifaces = {k: to_device(c, dev)
              for k, c in sample_interfaces(a.n_iface, a.seed).items()}

    ic_coll, psi0 = ic_targets()
    ic = (to_device(ic_coll, dev), psi0.to(dev))

    loss_fn = dict(bcs=bcs, ic=ic, ifaces=ifaces, mats=load_materials())
    return net, loss_fn, coll


def parse(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        fromfile_prefix_chars="@",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("network")
    g.add_argument("--layers", type=int, default=2,
                   help="2 reproduces the Step 3.4 measurement; 8 is full()")
    g.add_argument("--width", type=int, default=64)
    g.add_argument("--seed", type=int, default=20250812)
    g.add_argument("--device", default="cpu")

    g = ap.add_argument_group("sampling")
    g.add_argument("--n-pde", type=int, default=500)
    g.add_argument("--n-bc", type=int, default=500)
    g.add_argument("--n-iface", type=int, default=500)
    g.add_argument("--t-max", type=float, default=30.0)

    g = ap.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=1000)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--lr-schedule", choices=("cosine", "flat"), default="cosine",
                   help="cosine = roadmap three-phase. flat pins lr for "
                   "control runs that must stay comparable.")

    g = ap.add_argument_group("balancer (Step 3.4)")
    g.add_argument("--every", type=int, default=25,
                   help="update cadence. Raise to 100-200 if VRAM is tight; "
                        "peak memory is during the update, not steady state.")
    g.add_argument("--lam", type=float, default=0.3,
                   help="log-space EMA rate on the correction factor")
    g.add_argument("--warmup", type=int, default=0)
    g.add_argument("--no-balance", action="store_true",
                   help="unit weights. The control arm: under unit weights "
                        "the objective is ~99.9%% bc_mech and everything else "
                        "degrades. Not a training mode.")
    g = ap.add_argument_group("io")
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--ckpt-every", type=int, default=500)
    g.add_argument("--out", default="runs/dev")
    g.add_argument("--resume", metavar="CKPT")
    g.add_argument("--no-feedback", action="store_true",
                   help="Kozeny-Carman porosity-strain feedback OFF. The "
                        "one-way arm of the Step 6.2 ablation (Day 51).")
    g = ap.add_argument_group("L-BFGS polish (Step 4.1b)")
    g.add_argument("--lbfgs-epochs", type=int, default=0,
                   help="L-BFGS iterations AFTER the Adam phase. 0 (default) "
                        "= Adam only, which is every run before Day 27. "
                        "Each iteration costs up to --lbfgs-max-eval closure "
                        "evaluations, so 200 L-BFGS steps can cost as much as "
                        "4000 Adam steps.")
    g.add_argument("--lbfgs-lr", type=float, default=1.0,
                   help="L-BFGS step scale. 1.0 with strong_wolfe, which "
                        "chooses the step itself; lower only if the line "
                        "search reports failures.")
    g.add_argument("--lbfgs-history", type=int, default=50)
    g.add_argument("--lbfgs-max-eval", type=int, default=25,
                   help="closure evaluations per iteration (line search cap)")

    g = ap.add_argument_group("ansatz (NearPhysical)")
    g.add_argument("--eps-psi", type=float, default=0.3,
                   help="NearPhysical psi perturbation scale. 3e-3 (the "
                        "pre-Day-25 value) caps psi excursion at 0.5 m of "
                        "head, which contradicts the Tm rain flux BC.")
    g.add_argument("--ansatz", choices=("unbounded", "cap", "exp"),
                   default="unbounded",
                   help="how psi is held unsaturated. DEFAULT unbounded = the "
                        "shipped behaviour. At eps_psi = 0.3 that puts 15-35%% "
                        "of collocation points at psi >= 0 on an untrained "
                        "net, in a domain that is unsaturated everywhere by "
                        "construction. cap/exp remove that; which to adopt is "
                        "OPEN -- see DECISIONS.md D-A.1 and "
                        "scripts/ansatz_ablation.py.")
    g.add_argument("--cap-k", type=float, default=100.0,
                   help="sharpness of --ansatz cap. Larger = sharper = less "
                        "IC distortion but less of the spread fixed. k=20 "
                        "distorts psi_0 = -3 m by 145%% and is unusable; "
                        "k=1000 leaves the spread at 100x.")
    g.add_argument("--eps-uv", type=float, default=None,
                   help="ansatz displacement scale u*, v* = eps_uv * net. "
                        "Default None = the shipped 1e-3. THE FORMULA SEED "
                        "DEPENDS ON IT: w0[pde_richards] = "
                        "((Pi_M*eps_uv)/(Pi_R*eps_psi))^2, so this flag also "
                        "re-derives the seed. Use the flag; do not edit "
                        "weighting._EPS_UV, which changes the seed for every "
                        "run and every test at once (O-20).")
    g.add_argument("--seed-source", choices=("formula", "t0", "geomean"),
                   default="formula",
                   help="which D-W.7 seed to start the balancer from, at "
                        "layers>=8. formula = pi_seed(), derived from the Pi "
                        "groups and the ansatz prefactors, no measurement -- "
                        "the DEFAULT since Day 26. t0 and geomean are the "
                        "measured w^ columns, kept as ablation arms: "
                        "scripts/seed_variance.py shows w^[pde_richards] "
                        "moving 1264x across sampling seeds at N=1200, so "
                        "they are samples of a heavy tail rather than "
                        "constants. formula wins the matched three-arm "
                        "ablation on five of seven terms.")
    return ap.parse_args(argv)
   
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)

    net, loss_fn, coll = setup(a)
    print(net.summary())
    if a.device != "cpu" and str(net.device) == "cpu":
        print(f"ERROR: --device {a.device} requested but resolved to cpu. "
              f"torch.cuda.is_available()={torch.cuda.is_available()}",
              file=sys.stderr)
        return 2
    print(f"N_PDE={a.n_pde}  N_BC={a.n_bc}  N_iface={a.n_iface}  "
          f"device={net.device}")

    # D-W.7 at depth: pde_mech's gradient norm is 342,000x the anchor's at
    # 8x64 but 386x BELOW it at 2x64. The required w^ = 2.9e-06 is outside
    # the +/-3 order clip on c, so without the seed the balancer clamps and
    # leaves pde_mech ~340x over-weighted for the whole run.
    if a.layers >= 8:
        seed_w = {"t0": PI_SEED_8X64_T0,
                  "geomean": PI_SEED_8X64,
                  "formula": None}[a.seed_source]
    else:
        seed_w = None
    if a.seed_source == "formula" and getattr(a, "eps_uv", None) is not None:
        # The balancer would otherwise fall back to the module PI_SEED, which
        # is computed at the default eps_uv: a 10x eps_uv is a 100x seed.
        seed_w = pi_seed(eps_psi=a.eps_psi, eps_uv=a.eps_uv)
        print(f"formula seed re-derived at eps_uv={a.eps_uv:g}: "
              f"w0[pde_richards]={seed_w['pde_richards']:.4g}")
    print(f"seed source: {a.seed_source}"
          + ("" if seed_w is None else
             "  " + "  ".join(f"{k}={v:.3g}" for k, v in sorted(seed_w.items()))))
    bal = LossBalancer(TERMS, net.parameters(),
                       BalancerConfig(warmup=a.warmup, every=a.every,
                                      lam=a.lam, seed=seed_w))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    start, L0 = 0, None
    if a.resume:
        d = torch.load(a.resume, map_location=net.device, weights_only=False)
        net.load_state_dict(d["net"])
        opt.load_state_dict(d["opt"])
        bal.load_state_dict(d["bal"])       # raises if PI_SEED has changed
        prev_ep = d.get("cfg", {}).get("epochs")
        if prev_ep is not None and prev_ep != a.epochs:
            print(f"WARNING: resuming with --epochs {a.epochs} but the "
                  f"checkpoint was written under --epochs {prev_ep}. The lr "
                  f"schedule is a fraction of run length, so it will reshape "
                  f"mid-run.", file=sys.stderr)
        start, L0 = d["step"] + 1, d.get("L0")
        print(f"resumed {a.resume} at step {start}; "
              f"c = { {k: round(v, 3) for k, v in bal.c.items()} }")

    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=2)
    log = open(f"{a.out}/log.jsonl", "a")
    t0, n_done = time.time(), 0

    for step in range(start, a.epochs + 1):
        lr_now = a.lr * (lr_factor(step, a.epochs)
                         if a.lr_schedule == "cosine" else 1.0)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        L = losses_at(net, loss_fn, coll, feedback=not a.no_feedback)
        if L0 is None:
            L0 = {k: float(v.detach()) for k, v in L.items()}

        if not a.no_balance:
            bal.maybe_update(step, L)
            total = bal.total(L)
        else:
            total = sum(v for k, v in L.items() if "::" not in k)

        opt.zero_grad()
        total.backward()
        opt.step()
        n_done += 1
                  
        if step % a.log_every == 0:
            _write_log(log, step, "adam", total, L, L0, bal, net, coll,
                       lr_now, (time.time() - t0) / max(n_done, 1))

        if a.ckpt_every and step and step % a.ckpt_every == 0:
            _atomic_save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "bal": bal.state_dict(), "step": step, "L0": L0,
                        "cfg": vars(a)}, f"{a.out}/ckpt_{step:06d}.pt")

    # A checkpoint at the handover, ALWAYS, independent of --ckpt-every. Fig. 4
    # marks this step, and the Adam-only field has to remain recoverable: if
    # the polish makes things worse, the comparison is against this file.
    _atomic_save({"net": net.state_dict(), "opt": opt.state_dict(),
                  "bal": bal.state_dict(), "step": a.epochs, "L0": L0,
                  "cfg": vars(a)}, f"{a.out}/ckpt_adam_final.pt")

    if a.lbfgs_epochs > 0:
        n_done += _lbfgs_phase(a, net, bal, loss_fn, coll, L0, log, t0, n_done)

    _atomic_save({"net": net.state_dict(), "opt": opt.state_dict(),
                "bal": bal.state_dict(), "step": a.epochs + a.lbfgs_epochs,
                "L0": L0, "cfg": vars(a)}, f"{a.out}/ckpt_final.pt")
    log.close()

    print()
    print(bal.report())
    sec = (time.time() - t0) / max(n_done, 1)
    print(f"\n{sec:.4f} s/epoch over {n_done} epochs")
    for n in (10_000, 40_000, 100_000):
        print(f"  {n:>7,} epochs -> {sec * n / 3600:6.2f} h")
    if str(net.device).startswith("cuda"):
        print(f"\npeak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB "
              f"(reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB)")
        print("Peak is during bal.update() -- seven autograd.grad calls with "
              "retain_graph=True. Raise --every before cutting --n-pde.")
    return 0

def _lbfgs_phase(a, net, bal, loss_fn, coll, L0, log, t0, n_done):
    """Second-order polish after Adam. Returns the number of iterations run.

    WEIGHTS ARE FROZEN AT HANDOVER, and this is a correctness requirement
    rather than a tuning choice. L-BFGS builds its inverse-Hessian estimate
    from (s, y) pairs -- parameter and GRADIENT differences across iterations --
    and the strong-Wolfe line search compares objective values across several
    evaluations within a single iteration. If `bal.w` changed underneath, the
    curvature pairs would come from different objectives and the line search
    would compare values of different functions. Both are silent failures: the
    run would not error, it would just descend on nothing in particular. So the
    balancer is not stepped here, and `bal.total` is evaluated with whatever
    weights the Adam phase ended on.

    The consequence is worth stating in Methodology: the polished field is the
    minimiser of a FIXED weighted objective, the one GradNorm arrived at by
    step `--epochs`. It is not the minimiser of the adaptively-weighted
    problem, because that problem has no fixed objective to minimise.

    A WARNING ABOUT THE CURRENT STATE OF THE MODEL. As of Day 27 a 2000-epoch
    Adam run ends with `w[pde_richards]` around 8.3e+03 on a term whose loss is
    pinned at 1e-05 of its start -- not because it is solved but because the
    field has drifted to ~163 m of suction where every flux term underflows
    (`docs/open_items.md`, Day 27; `activity_floor` cannot distinguish the two
    cases). Freezing those weights and applying a second-order method drives
    the field further into that regime, faster and more precisely. Convergence
    will look excellent. Check `psi_max_m` and `bc` in the log before believing
    any of it.
    """
    dev_pin = list(net.parameters())
    opt = torch.optim.LBFGS(dev_pin, lr=a.lbfgs_lr,
                            max_iter=1, max_eval=a.lbfgs_max_eval,
                            history_size=a.lbfgs_history,
                            tolerance_grad=0, tolerance_change=0,
                            line_search_fn="strong_wolfe")

    frozen_w = dict(bal.w)
    print(f"\n--- L-BFGS handover at step {a.epochs} "
          f"({a.lbfgs_epochs} iterations, weights frozen) ---")
    print("  w = " + "  ".join(f"{k}={v:.4g}" for k, v in
                               sorted(frozen_w.items())))

    state = {"L": None, "total": None, "nfev": 0}

    def closure():
        opt.zero_grad()
        L = losses_at(net, loss_fn, coll, feedback=not a.no_feedback)
        total = bal.total(L)          # frozen weights; bal is never stepped
        total.backward()
        state["L"], state["total"] = L, total
        state["nfev"] += 1
        return total

    done = 0
    for it in range(1, a.lbfgs_epochs + 1):
        step = a.epochs + it
        opt.step(closure)
        done += 1
        if state["L"] is None:        # line search made no evaluation at all
            print(f"L {step:6d}  line search produced no evaluation; stopping")
            break
        if it % a.log_every == 0 or it == a.lbfgs_epochs:
            _write_log(log, step, "lbfgs", state["total"], state["L"], L0,
                       bal, net, coll, a.lbfgs_lr,
                       (time.time() - t0) / max(n_done + done, 1))
        if not torch.isfinite(state["total"]):
            print(f"L {step:6d}  total is not finite; stopping")
            break

    print(f"\n  {state['nfev']} closure evaluations over {done} L-BFGS "
          f"iterations ({state['nfev'] / max(done, 1):.1f} per iteration; "
          f"the cap is --lbfgs-max-eval {a.lbfgs_max_eval})")
    return done


def _write_log(log, step, phase, total, L, L0, bal, net, coll, lr_now, sec):
    """One log record, shared by the Adam and L-BFGS phases.

    Factored out so the two phases cannot drift into different schemas --
    `tests/test_log_schema.py` pins the key set, and a second inline copy is
    exactly how the `clipped` field went missing once before.
    """
    raw = {k: float(v.detach()) for k, v in L.items()}

    # `spread` and `grad_norms` are only meaningful while the balancer is
    # being stepped. In the L-BFGS phase the weights are frozen and
    # bal._last_norms still holds whatever the final Adam update measured, so
    # reporting them would draw a flat line in Fig. 4 that looks like a
    # measurement and is a stale cache. Emit nulls instead.
    if phase == "adam":
        g = bal._last_norms
        live = [bal.w[k] * g[k] for k in TERMS
                if k not in bal._frozen and g.get(k, 0.0) > 0]
    else:
        g, live = {}, []

    # Day 27: the quantity the ansatz decision turns on. Section 5 is
    # unsaturated everywhere (Z_WT = 197 m is below Z_BASE = 200 m), so any
    # psi >= 0 is unphysical, and it is where van Genuchten K_r goes
    # near-singular and the Richards gradient becomes untrustable. Cheap: one
    # forward pass on the existing interior set, no grad.
    with torch.no_grad():
        _psi = net(coll.x, coll.z, coll.t)[:, 0:1] * SCALES.H_ref

    rec = {"step": step, "phase": phase, "total": float(total.detach()),
           "L": raw,
           "psi_sat_frac": float((_psi >= 0).double().mean()),
           "psi_max_m": float(_psi.max()),
           "psi_p50_m": float(_psi.median()),
           "L0": L0, "w": dict(bal.w), "c": dict(bal.c),
           "grad_norms": dict(g),
           "spread": (max(live) / min(live)) if len(live) > 1 else None,
           "clipped": sorted(bal._clipped),
           "log_c_raw": dict(bal._log_c_raw),
           "frozen": sorted(bal._frozen),
           "lr": lr_now,
           "sec_per_epoch": sec}
    log.write(json.dumps(rec) + "\n")
    log.flush()

    sp = rec["spread"]
    tag = "" if phase == "adam" else "L "
    print(f"{tag}{step:6d}  " + "  ".join(
        f"{k}={raw[k] / L0[k]:.3g}" for k in
        ("bc_mech", "pde_mech", "pde_richards", "bc", "ic_head", "ic_disp"))
        + (f"   spread={sp:.3g}" if sp else "")
        + f"   {sec:.3f}s/ep")


def lr_factor(step, total, warm=0.10, decay_end=0.50, floor=0.10):
    """Multiplier on base lr. Roadmap Step 4.1 schedule as fractions of run
    length: flat warm-up, cosine decay, flat refinement. Pure function of step,
    so resume needs no scheduler state."""
    w, d = warm * total, decay_end * total
    if step < w:
        return 1.0
    if step < d:
        p = (step - w) / (d - w)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
    return floor    

if __name__ == "__main__":
    sys.exit(main())