"""What did a run actually learn? Loss values cannot distinguish a
converged field from a collapsed one -- pde_richards at 5e-08 is either
excellent or a sign psi went trivial. This looks at the fields.

    python scripts/inspect_prod01.py                        # prod01, as before
    python scripts/inspect_prod01.py runs/prod02_smoke/ckpt_002000.pt

EPS_PSI IS NOT OPTIONAL (Day 26). `NearPhysical`'s default moved 3e-3 -> 0.3
on Day 25, and this script rebuilds the wrapper before loading weights. Reading
a prod01 checkpoint under the new default reports psi* a hundred times larger
than the network was trained to emit, which is the opposite of the collapse
this script exists to detect. Pass --eps-psi for anything trained before 2 Sep.
"""
import argparse

import torch
from src.model import PINN, NearPhysical
from src.loss import psi_of, uv_of
from src.config import full, bounds_from_geometry
from src.nondim import SCALES

BOUNDS = bounds_from_geometry(t_max=30.0, s=SCALES)
print("bounds_from_geometry:", BOUNDS)

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("ckpt", nargs="?", default="runs/prod01/ckpt_014000.pt")
ap.add_argument("--eps-psi", type=float, default=None,
                help="ansatz prefactor the checkpoint was TRAINED with. "
                     "Default: NearPhysical's current default (0.3). "
                     "prod01 and anything else before 2 Sep needs 3e-3.")
a = ap.parse_args()

d = torch.load(a.ckpt, map_location="cpu", weights_only=False)
print("checkpoint:", a.ckpt, "  step:", d.get("step"))
print("ckpt dtype:", next(iter(d["net"].values())).dtype)

cfg = full()
cfg.device = "cpu"
cfg.dtype  = "float64"
kw = {} if a.eps_psi is None else {"eps_psi": a.eps_psi}
net = NearPhysical(PINN(cfg, BOUNDS), **kw)
print("ansatz:", net.summary())
net.load_state_dict(d["net"])
net.eval()

# What the net was ACTUALLY trained on. Buffers live in the state_dict and
# load_state_dict overwrites whatever we constructed with, so these win.
KEYS = ("x_min", "x_max", "z_min", "z_max", "t_min", "t_max")
ckpt_b = {k.split(".")[-1]: float(v)
          for k, v in d["net"].items() if k.split(".")[-1] in KEYS}
print("checkpoint buffers:", ckpt_b)
B = ckpt_b if len(ckpt_b) == 6 else BOUNDS   # prefer the checkpoint

# Starred coordinates. L_ref = 170 m, so x in [-0.4, 268.0] m,
# z in [199.5, 358.0] m. t* == days because T_ref = 86400 s.
X0, X1 = B["x_min"], B["x_max"]
Z0, Z1 = B["z_min"], B["z_max"]

L, H, U = SCALES.L_ref, SCALES.H_ref, SCALES.U_ref
print(f"L_ref={L} m  H_ref={H} m  U_ref={U:.4g} m")
print(f"physical extents: x {X0*L:.1f}..{X1*L:.1f} m   "
      f"z {Z0*L:.1f}..{Z1*L:.1f} m")

fields = {}
for t_days in [0.0, 1.0, 7.0, 30.0]:
    xs = torch.linspace(X0, X1, 40, dtype=torch.float64)
    zs = torch.linspace(Z0, Z1, 20, dtype=torch.float64)
    X, Z = torch.meshgrid(xs, zs, indexing="ij")
    x = X.reshape(-1, 1)
    z = Z.reshape(-1, 1)
    t = torch.full_like(x, t_days)
    with torch.no_grad():
        out = net(x, z, t)          # three separate (N,1) leaves
    psi = psi_of(out)
    uv = uv_of(out)
    fields[t_days] = (psi.clone(), uv.clone(), z.clone())

    print(f"t={t_days:5.1f} d  "
          f"psi*: min {psi.min():9.3e}  max {psi.max():9.3e}  "
          f"mean {psi.mean():9.3e}  |  "
          f"|u*|max {uv[:, 0].abs().max():9.3e}  "
          f"|v*|max {uv[:, 1].abs().max():9.3e}  |  "
          f"psi {psi.min()*H:8.2f}..{psi.max()*H:8.2f} m  "
          f"|v| {uv[:, 1].abs().max()*U*1000:9.3e} mm")

# --- the two tests that actually decide the branch ---------------------
psi0, uv0, z0 = fields[0.0]
psi30, uv30, _ = fields[30.0]

print(f"\nmax |psi(30) - psi(0)|  = {(psi30 - psi0).abs().max():.3e} starred"
      f"  ({(psi30 - psi0).abs().max()*H:.3f} m)")
print(f"max |v(30)  - v(0)|     = "
      f"{(uv30[:, 1] - uv0[:, 1]).abs().max():.3e} starred")

# t=0 should reproduce psi_initial(z) = -(z - 197) exactly, i.e.
# psi* = -(L*z* - 197)/H.  Unless NearPhysical hard-imposes the IC,
# in which case this is satisfied by construction and proves nothing.
psi_ic = -(L * z0 - 197.0) / H
print(f"IC residual at t=0: max |psi* - psi*_analytic| = "
      f"{(psi0.reshape(-1, 1) - psi_ic).abs().max():.3e}"
      f"   (expected psi* range {psi_ic.min():.3f} .. {psi_ic.max():.3f})")


# --- ansatz check: what does an UNTRAINED net give on the same grid? ------
# x, z, t still hold the t = 30 d grid from the loop above.
print("\n(comparison grid: t = %.1f d)" % t[0].item())

net_r = NearPhysical(PINN(cfg, BOUNDS), **kw)  # fresh init, no load_state_dict
net_r.eval()
with torch.no_grad():
    out_r = net_r(x, z, t)
psi_r, uv_r = psi_of(out_r), uv_of(out_r)

print(f"random-init psi*: min {psi_r.min():9.3e}  max {psi_r.max():9.3e}  "
      f"mean {psi_r.mean():9.3e}")
print(f"trained     psi*: min {psi30.min():9.3e}  max {psi30.max():9.3e}  "
      f"mean {psi30.mean():9.3e}")
print(f"random-init |u*|max {uv_r[:, 0].abs().max():9.3e}  "
      f"|v*|max {uv_r[:, 1].abs().max():9.3e}")
print(f"trained     |u*|max {uv30[:, 0].abs().max():9.3e}  "
      f"|v*|max {uv30[:, 1].abs().max():9.3e}")