"""
smoke_test.py — Day 22, first full-size run on the lab PC.

Purpose: NOT convergence. Three questions only.
  1. Does the full config (8x64, N_PDE=10,000) fit in VRAM, and how much headroom?
  2. What is the true seconds/epoch, and what does that make the full run cost?
  3. Is the loss curve the right SHAPE in the first 2,000 epochs?

Run:
    cd <repo root>
    python smoke_test.py                 # full ladder + 2,000-epoch run
    python smoke_test.py --ladder-only   # just the VRAM probe, ~30 s
    python smoke_test.py --epochs 500    # shorter

Outputs (into runs/smoke_<timestamp>/):
    env.json          environment fingerprint — paste this to your supervisor
    ladder.csv        peak VRAM vs N_PDE
    loss.csv          per-epoch, per-term losses + adaptive weights
    summary.txt       the three answers

Nothing here writes a checkpoint. This run is disposable by design.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import statistics
import sys
import time

import torch

# ---------------------------------------------------------------------------
# CRITICAL: disable TF32 before anything touches the GPU.
#
# On Ampere and later (RTX 30xx/40xx, A-series), PyTorch silently routes fp32
# matmuls through TF32, which carries a 10-bit mantissa. For a PINN taking
# SECOND derivatives of tanh through 8 layers, that truncation shows up as a
# noise floor in d2psi/dx2 that no amount of training removes. It is the most
# likely reason a run behaves on the laptop (float64, CPU) and misbehaves on
# the lab PC. Turn it off here; measure the cost; decide deliberately.
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ===========================================================================
# PASTE 1 of 2 — replace the whole ADAPT ME block in smoke_test.py
# (from `from src.config import ...` down to the end of build_trainer)
# ===========================================================================
import torch
 
from src.config import BOUNDS, full, resolve_device
from src.model import PINN
from src.loss import total_loss, ic_targets
from src.materials import load_materials
from src.sampling import (sample_interior, sample_boundary,
                          sample_interfaces, to_device)
from src.sigma0 import attach_sigma0
 
DTYPES = {"float32": torch.float32, "float64": torch.float64}
 
# Held fixed across the ladder; only N_PDE varies. Roadmap budget:
# N_BC 2,000-5,000, N_IC 2,000 (the cache is 4,000), interfaces to taste.
SEED = 20250812
T_MAX = 30.0
N_BC = 3_000
N_IFACE = 2_000
 
# Production configuration. Flip these to measure what each half costs:
#   mechanics off  -> hydraulic-only baseline
#   feedback off   -> the one-way run you need for the Step 6.2 ablation
INCLUDE_MECHANICS = True
INCLUDE_FEEDBACK = True
 
 
def build_trainer(cfg, n_pde: int):
    """Assemble model + one-step closure at the given N_PDE.
 
    Deliberately runs with STATIC DEFAULT_WEIGHTS, no LossBalancer. Day 22 is
    three questions: does it fit, how fast is it, what shape is the curve.
    GradNorm is a Day 23 question, and leaving it out means the seconds/epoch
    number measures your physics rather than the balancer's rebalance passes.
    Add it once you have the static baseline to compare against.
    """
    device = resolve_device(cfg.device)
    dtype = DTYPES[cfg.dtype]
 
    # --- sample -----------------------------------------------------------
    coll = sample_interior(n_pde, SEED, t_max=T_MAX, dtype=dtype)
    bcs = sample_boundary(N_BC, SEED, T_MAX)
    ifaces = sample_interfaces(N_IFACE, SEED, T_MAX)
    ic = ic_targets(device=device)
    mats = load_materials()
 
    # --- sigma_0 BEFORE the device move -----------------------------------
    # attach_sigma0 reads the FE cache written by 17_sigma0_solve.py, which is
    # host-side. Attach on CPU, then move the assembled bundle in one go.
    # L_BC_mech needs it on the boundaries too: traction-free is
    # sigma_total . n = 0, and sigma_total starts from sigma_0.
    if INCLUDE_MECHANICS:
        coll = attach_sigma0(coll)
        bcs = {k: attach_sigma0(v) for k, v in bcs.items()}
 
    # --- move -------------------------------------------------------------
    coll = to_device(coll, device)
    bcs = {k: to_device(v, device) for k, v in bcs.items()}
    ifaces = {k: to_device(v, device) for k, v in ifaces.items()}
 
    # --- model + optimiser ------------------------------------------------
    model = PINN(cfg, BOUNDS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
 
    def step_fn():
        opt.zero_grad(set_to_none=True)
        total, parts = total_loss(
            model, coll, bcs, ic, ifaces, mats,
            include_mechanics=INCLUDE_MECHANICS,
            include_feedback=INCLUDE_FEEDBACK,
            per_term=True,
        )
        total.backward()
        opt.step()
        # parts are already detached; float() is safe and costs one sync.
        return {k: float(v) for k, v in parts.items()}
 
    return model, step_fn, list(model.parameters())
# ===========================================================================


def fingerprint() -> dict:
    d = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "python": sys.version.split()[0],
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d |= {
            "gpu": p.name,
            "vram_total_gb": round(p.total_memory / 2**30, 2),
            "sm_count": p.multi_processor_count,
            "capability": f"{p.major}.{p.minor}",
        }
    return d


@contextlib.contextmanager
def peak_memory():
    """Yields a dict filled with peak allocated/reserved GB on exit.

    RESERVED is the number that decides OOM — it is what the caching allocator
    has actually taken from the driver. Allocated is always smaller and will
    flatter you.
    """
    out = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        yield out
    finally:
        torch.cuda.synchronize()
        out["alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
        out["reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30


def timed_steps(step_fn, n: int) -> list[float]:
    """Per-step wall time, synchronised.

    CUDA launches are ASYNCHRONOUS. Without synchronize() you measure how fast
    Python can queue kernels, not how fast they run — which is why naive PINN
    timings come out ~10x too optimistic and the wall-clock estimate collapses
    on contact with reality.
    """
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def vram_ladder(cfg, sizes, log) -> list[dict]:
    """One step at each N_PDE. Finds the ceiling in 30 s instead of at epoch 1800."""
    rows = []
    for n in sizes:
        try:
            model, step_fn, _ = build_trainer(cfg, n)
            step_fn()                      # warm-up: allocator + cuDNN autotune
            with peak_memory() as m:
                dt = timed_steps(step_fn, 5)
            row = {
                "n_pde": n,
                "ok": True,
                "reserved_gb": round(m["reserved_gb"], 3),
                "alloc_gb": round(m["alloc_gb"], 3),
                "s_per_epoch": round(statistics.median(dt), 5),
            }
            del model, step_fn
        except torch.cuda.OutOfMemoryError:
            row = {"n_pde": n, "ok": False, "reserved_gb": None,
                   "alloc_gb": None, "s_per_epoch": None}
        torch.cuda.empty_cache()
        rows.append(row)
        log(f"  N_PDE={n:>6,}  " + ("OOM" if not row["ok"] else
            f"reserved {row['reserved_gb']:.2f} GB   {row['s_per_epoch']*1e3:.1f} ms/epoch"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--n-pde", type=int, default=10_000)
    ap.add_argument("--warmup", type=int, default=50,
                    help="Steps excluded from timing (allocator + autotune).")
    ap.add_argument("--ladder-only", action="store_true")
    ap.add_argument("--adam-epochs", type=int, default=100_000,
                    help="Planned Stage-1 length, for the extrapolation.")
    ap.add_argument("--lbfgs-iters", type=int, default=20_000)
    ap.add_argument("--lbfgs-closures", type=float, default=2.0,
                    help="Function evals per L-BFGS iteration under strong-Wolfe.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("No CUDA device. This script is for the lab PC, not WhiteShadow.")

    out = pathlib.Path("runs") / f"smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    logfile = (out / "summary.txt").open("w")

    def log(msg=""):
        print(msg)
        logfile.write(msg + "\n")
        logfile.flush()

    env = fingerprint()
    (out / "env.json").write_text(json.dumps(env, indent=2))
    log(f"GPU        : {env['gpu']}  ({env['vram_total_gb']} GB, cc {env['capability']})")
    log(f"torch      : {env['torch']}  CUDA {env['cuda_runtime']}   TF32 off")

    cfg = full()
    log(f"config     : {cfg.n_layers}x{cfg.n_neurons} {cfg.dtype} on {cfg.device}")
    log()

    # ---- 1. VRAM ladder ---------------------------------------------------
    log("VRAM ladder (1 step each)")
    rows = vram_ladder(cfg, [10_000, 20_000, 50_000], log)
    with (out / "ladder.csv").open("w") as f:
        f.write("n_pde,ok,reserved_gb,alloc_gb,s_per_epoch\n")
        for r in rows:
            f.write(f"{r['n_pde']},{r['ok']},{r['reserved_gb']},"
                    f"{r['alloc_gb']},{r['s_per_epoch']}\n")

    base = next((r for r in rows if r["n_pde"] == args.n_pde and r["ok"]), None)
    if base is None:
        log(f"\nFAIL: N_PDE={args.n_pde:,} does not fit. Stop and reduce before Day 23.")
        return
    headroom = env["vram_total_gb"] - base["reserved_gb"]
    log(f"\n  headroom at N_PDE={args.n_pde:,}: {headroom:.2f} GB "
        f"({100*headroom/env['vram_total_gb']:.0f}% of card)")
    if args.ladder_only:
        return

    # ---- 2. Timed 2,000-epoch run ----------------------------------------
    log(f"\nRunning {args.epochs:,} epochs at N_PDE={args.n_pde:,} "
        f"(first {args.warmup} excluded from timing)")
    model, step_fn, _ = build_trainer(cfg, args.n_pde)
    log(f"  {model.summary()}")

    keys = ["total",
            "contrib_pde", "contrib_bc", "contrib_ic", "contrib_interface",
            "contrib_pde_mech", "contrib_bc_mech"]
    csv = (out / "loss.csv").open("w")
    csv.write("epoch,s_per_epoch," + ",".join(keys) + "\n")

    times, first_bad = [], None
    with peak_memory() as mem:
        for ep in range(args.epochs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            terms = step_fn()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if ep >= args.warmup:
                times.append(dt)

            vals = [terms.get(k, float("nan")) for k in keys]
            csv.write(f"{ep},{dt:.6f}," + ",".join(f"{v:.6e}" for v in vals) + "\n")

            if first_bad is None and not (terms["total"] == terms["total"]
                                          and abs(terms["total"]) != float("inf")):
                first_bad = ep
                log(f"  !! non-finite total loss at epoch {ep} — stopping")
                break

            if ep % 100 == 0:
                csv.flush()
                log(f"  ep {ep:>5}  total {terms['total']:.3e}  "
                    f"pde {terms.get('contrib_pde', float('nan')):.3e}  "
                    f"mech {terms.get('contrib_pde_mech', float('nan')):.3e}  "
                    f"bc {terms.get('contrib_bc', float('nan')):.3e}  "
                    f"ic {terms.get('contrib_ic', float('nan')):.3e}")
    csv.close()

    # ---- 3. Extrapolation -------------------------------------------------
    s = statistics.median(times)
    p95 = sorted(times)[int(0.95 * len(times)) - 1]
    adam_h = args.adam_epochs * s / 3600
    lbfgs_h = args.lbfgs_iters * args.lbfgs_closures * s / 3600

    log()
    log("=" * 62)
    log(f"seconds/epoch      median {s*1e3:.1f} ms   p95 {p95*1e3:.1f} ms")
    log(f"peak VRAM          {mem['reserved_gb']:.2f} / {env['vram_total_gb']} GB reserved")
    log(f"projected Adam     {args.adam_epochs:,} ep  ->  {adam_h:.2f} h")
    log(f"projected L-BFGS   {args.lbfgs_iters:,} it  ->  {lbfgs_h:.2f} h  "
        f"(at {args.lbfgs_closures} closures/iter)")
    log(f"FULL RUN           ~{adam_h + lbfgs_h:.1f} h")
    log(f"  x3 cycles (Phase 4 realistic)  ~{3*(adam_h + lbfgs_h):.1f} h")
    log(f"  x27 SSR sweep (Phase 5, warm-started at ~1/3 cost)  "
        f"~{27*(adam_h + lbfgs_h)/3:.1f} h")
    log("=" * 62)
    if p95 > 2 * s:
        log("NOTE: p95 is >2x median — something is stalling. Check for CPU-side "
            "work in the loss (per-material Python loops, .item() calls, host "
            "syncs) before trusting the projection.")
    logfile.close()
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()