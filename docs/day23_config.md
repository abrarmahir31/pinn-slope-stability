# Day 23 — Lab PC production configuration

Hardware: NVIDIA GeForce RTX 4060 Ti, 16.0 GB, cc 8.9.
torch 2.13.0+cu130, CUDA 13.0, TF32 disabled, float64.
Every number below is measured on this machine, not estimated.

Production command:

    set PYTHONPATH=.
    python scripts/train.py --layers 8 --n-pde 10000 --n-bc 3000 --n-iface 500 ^
        --epochs 100000 --lr 1e-3 --lr-schedule cosine ^
        --every 25 --lam 0.3 --warmup 0 ^
        --device cuda --log-every 100 --ckpt-every 500 --out runs/prod01

---

## N_PDE = 10,000

| N_PDE | peak reserved | s/epoch | 100k Adam |
|-------|---------------|---------|-----------|
| 10,000 | 4.26 GB | 0.576 | 15.99 h |
| 20,000 | 7.79 GB | 0.915 | 25.42 h |
| 50,000 | 19.41 GB | 4.92 | — |

50,000 is **not** a fit. 19.41 GB on a 16 GB card means the Windows driver
spilled to host RAM over PCIe rather than raising OutOfMemoryError. The timing
confirms it: 10k→20k scales sublinearly (1.6x for 2x points, launch overhead
amortising), but 20k→50k costs 5.7x for 2.5x points. The extra ~2,800 ms is
PCIe traffic, not compute.

Set NVIDIA Control Panel → Manage 3D Settings → *CUDA — Sysmem Fallback
Policy* → **Prefer No Sysmem Fallback**, so a future OOM is honest rather than
merely slow.

10,000 chosen over 20,000: +63% wall-clock (≈9.4 h per cycle, ≈28 h across
Phase 4's three cycles) for **no measured accuracy benefit**. There is
currently no evidence that 10,000 under-resolves the wetting front. Revisit
only if the Phase 4 diagnosis at Day 28 shows spatial under-resolution, or if
L-BFGS plateaus at 1e-4 (the roadmap's stated symptom of an inadequate
collocation budget).

Peak VRAM occurs during `bal.update()` — seven `autograd.grad` calls with
`retain_graph=True` — not in steady state. Measured with the balancer active,
so 4.26 GB is the true ceiling, leaving ~11.7 GB headroom.

## Learning rate: cosine, three-phase

Implemented as `lr_factor(step, total)` in `scripts/train.py`, expressed as
fractions of run length (warm 0.10, decay_end 0.50, floor 0.10). At
`--epochs 100000` this reproduces the roadmap Step 4.1 spec exactly:

| step | factor | lr |
|------|--------|-----|
| 0 – 10,000 | 1.0 | 1e-3 flat warm-up |
| 20,000 | 0.868 | 8.7e-4 |
| 30,000 | 0.550 | 5.5e-4 (cosine midpoint) |
| 50,000 | 0.100 | 1e-4 |
| 50,000 – 100,000 | 0.100 | 1e-4 flat refinement |

Deliberately **not** a `torch.optim.lr_scheduler` object. `lr_now` is a pure
function of `step`, so a run resumed from a checkpoint picks up at exactly the
right value with no scheduler state to save or restore. This matters: load
shedding makes interruption routine, and a mis-restored scheduler would be
silent.

**Known hazard.** The schedule is a fraction of `--epochs`. Resuming with a
different `--epochs` than the original run reshapes it mid-flight. Guard added
to the resume block; pass the same `--epochs` on resume.

## GradNorm warm-up = 0

Tested at 500 epochs, N_PDE = 10,000, cosine schedule:

| warmup | spread | clipped | frozen | pde_richards |
|--------|--------|---------|--------|--------------|
| 0 | 71.28 | 3 | 2 | 0.00178 |
| 100 | 78.37 | 3 | 2 | 0.00177 |
| 250 | 60.28 | 3 | 2 | 0.00177 |

`pde_richards` is identical to three significant figures. Spread varies
non-monotonically with no trend — noise, not signal. **Warm-up is inert here**,
because `PI_SEED_8X64` already corrects the initialisation imbalance a warm-up
would otherwise address: `pde_mech`'s gradient norm is 342,000x the anchor's at
8x64, and the required w = 2.9e-06 lies outside the ±3-order clip on c.

Recording an inert knob is itself the result. Set 0 and stop thinking about it.

Balancer cadence `--every 25` retained. Its cost is ~5% of wall-clock
(0.5625 s/epoch balancer-free vs 0.576 with), and peak VRAM at 10,000 is
unchanged. No reason to raise it.

---

## Open items this configuration does not resolve

**`pde_mech` clipping.** In the 2,000-epoch smoke run it began clipping at step
1550 and stayed clipped to the end — new, undocumented behaviour. It appears as
the mechanical residual falls four orders and the balancer wants to push its
weight below the −3-order floor. This is the clip becoming binding through
*success*, not failure. Watch across the first production run: if `pde_mech`
stays pinned for most of 100k epochs, it is over-weighted for most of training
and the clip bound needs revisiting.

**`interface` frozen.** Correct at t = 0 and asserted as such in
`grad_norm_table.py:34` — the hydrostatic IC gives zero Darcy flux everywhere
(`14_ic_checks.py`: total head uniform, ptp 0.00e+00 m), so the flux jump is
identically zero on both sides regardless of the three-order K contrast across
the top-of-Tm contact.

`scripts/check_iface_reach.py` measures whether it can ever activate:
**42 of 1000 contact points (4.2%) lie within the ~1–2 m diffusion length over
t_max = 30 d** (38/500 on mk_tm, 4/500 on mkd_tm). The remainder sit a median
10–17 m from any active boundary and will see zero flux for the whole
simulation. The term is therefore *enforced across the full contact but
exercised only near the toe*. That belongs in the methodology rather than being
left as an unexplained 1e-26 in the loss table.

Worth keeping rather than freezing out: the toe is where the failure surface
exits, so the 4.2% that can activate are in the most mechanically consequential
place.

**Confound warning — do not misread the 500-epoch table.** `warm0_flat` reached
spread 6.03 against `warm0`'s 71.28, which looks like strong evidence against
the cosine schedule. It is not. At `--epochs 500` the fraction-based schedule
puts steps 250–500 at lr = 1e-4 while the flat run held 1e-3 throughout, so the
flat run simply did ~10x more optimisation in the back half. A schedule defined
as a fraction of run length cannot be evaluated at 1/200th of production length.
Cosine is retained on the roadmap's authority; if it needs testing, the test is
a 5,000-epoch pair, not a 500-epoch one.

**E_ref.** `nondim.py` still carries 1e9 against the Phase-1 decision to use
100 MPa. Not implicated by anything measured here — the balancer handles the
resulting magnitude difference correctly — but still open from Phase 1 and
should be settled before the frozen baseline at Day 36–37.

## Operational notes

Wall-clock: Adam 100k ≈ 16.0 h, plus L-BFGS 20k iterations at ~2 closures
≈ 6.4 h. **~22 h per full cycle**, ~66 h across Phase 4's three cycles.

Checkpoints are ~735 KB and written atomically (`_atomic_save`, tmp + os.replace)
every 500 epochs — 4.8 minutes of exposure per interruption, at negligible cost.
Payload carries net, optimiser (Adam moments), balancer state, step and L0.

Load shedding: resume with

    python scripts/train.py --resume runs/prod01/ckpt_XXXXXX.pt --epochs 100000 --out runs/prod01

passing the **same** `--epochs`. Note `log.jsonl` opens in append mode, so a
resume concatenates rather than overwrites; filter by `step` when plotting.