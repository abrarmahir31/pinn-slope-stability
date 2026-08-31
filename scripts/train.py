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

HOW TO READ THE LOG -- the one thing that will mislead you
-----------------------------------------------------------
NOT the weighted total. It STEPS at every weight update. That is arithmetic,
not divergence: sum_i w_i L_i changes discontinuously when w changes, and a
run that is working perfectly shows a sawtooth. Anyone watching `total` for a
smooth decrease will conclude the run is broken when it is not.

Read the per-term UNWEIGHTED losses. Those are printed as ratios to their own
step-0 value (the columns) and stored raw in log.jsonl under "L".

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
from src.sigma0 import attach_sigma0
from src.weighting import (PI_SEED_8X64, BalancerConfig, LossBalancer, TERMS)
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
    net = NearPhysical(PINN(cfg, BOUNDS))
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
    seed_w = PI_SEED_8X64 if a.layers >= 8 else None
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
        start, L0 = d["step"] + 1, d.get("L0")
        print(f"resumed {a.resume} at step {start}; "
              f"c = { {k: round(v, 3) for k, v in bal.c.items()} }")

    json.dump(vars(a), open(f"{a.out}/config.json", "w"), indent=2)
    log = open(f"{a.out}/log.jsonl", "a")
    t0, n_done = time.time(), 0

    for step in range(start, a.epochs + 1):
        L = losses_at(net, loss_fn, coll)
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
            raw = {k: float(v.detach()) for k, v in L.items()}
            g = bal._last_norms
            live = [bal.w[k] * g[k] for k in TERMS
                    if k not in bal._frozen and g.get(k, 0.0) > 0]
            sec = (time.time() - t0) / max(n_done, 1)
            rec = {"step": step, "total": float(total.detach()), "L": raw,
                   "L0": L0, "w": dict(bal.w), "c": dict(bal.c),
                   "grad_norms": dict(g),
                   "spread": (max(live) / min(live)) if len(live) > 1 else None,
                   "clipped": sorted(bal._clipped),
                   "frozen": sorted(bal._frozen),
                   "sec_per_epoch": sec}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            sp = rec["spread"]
            print(f"{step:6d}  " + "  ".join(
                f"{k}={raw[k] / L0[k]:.3g}" for k in
                ("bc_mech", "pde_mech", "pde_richards", "bc",
                 "ic_head", "ic_disp"))
                + f"   spread={sp:.3g}" if sp else ""
                + f"   {sec:.3f}s/ep")

        if a.ckpt_every and step and step % a.ckpt_every == 0:
            _atomic_save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "bal": bal.state_dict(), "step": step, "L0": L0,
                        "cfg": vars(a)}, f"{a.out}/ckpt_{step:06d}.pt")

    _atomic_save({"net": net.state_dict(), "opt": opt.state_dict(),
                "bal": bal.state_dict(), "step": a.epochs, "L0": L0,
                "cfg": vars(a)}, f"{a.out}/ckpt_final.pt")
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


if __name__ == "__main__":
    sys.exit(main())