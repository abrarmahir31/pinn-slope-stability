"""Verify that dtheta_ref propagates to the Richards Pi groups and to L_PDE.

dtheta_ref enters twice: once through Pi_R_diff / Pi_R_grav (nondim.py:67,71)
and once through C_star (materials.py:84). Scaling it by k should scale both
Pi groups by 1/k and the loss by 1/k^2.

Run:  python scripts/verify_dtheta_propagation.py > docs/dtheta_ref_propagation.txt
"""

from src.config import BOUNDS, tiny
from src.loss import L_PDE
from src.materials import load_materials
from src.model import PINN
from src.nondim import Scales
from src.sampling import sample_interior

coll = sample_interior(200)
mats = load_materials()

print("dtheta_ref propagation check")
print("=" * 60)

results = {}
for dth in (0.33, 0.377):
    s = Scales(dtheta_ref=dth)
    net = PINN(tiny(), BOUNDS)
    L, _ = L_PDE(net, coll, mats, s=s)
    results[dth] = (s.Pi_R_diff, s.Pi_R_grav, float(L))
    print(f"dtheta_ref = {dth}")
    print(f"  Pi_R_diff = {s.Pi_R_diff:.6e}")
    print(f"  Pi_R_grav = {s.Pi_R_grav:.6e}")
    print(f"  L_PDE     = {float(L):.6e}")

k = 0.33 / 0.377
print("=" * 60)
print(f"k = 0.33/0.377 = {k:.6f}")
print(f"Pi_R_diff ratio = {results[0.377][0]/results[0.33][0]:.6f}  (expect {k:.6f})")
print(f"Pi_R_grav ratio = {results[0.377][1]/results[0.33][1]:.6f}  (expect {k:.6f})")
print(f"L_PDE ratio     = {results[0.377][2]/results[0.33][2]:.6f}  (expect {k**2:.6f})")