"""Can the interface flux-jump term activate within t_max?

Diffusion length over 30 days is ~1-2 m in the marl (D ~ 1e-6 m2/s from the
916-year diffusive timescale over L_ref = 170 m). A contact point further than
that from any active boundary sees zero flux for the whole simulation, so the
jump stays identically zero. This measures how many points are in reach.
"""
import numpy as np

import src.step21_geometry.geometry as g
from src.sampling import sample_interfaces
from src.nondim import SCALES as S, redim_x
from src.sampling import sample_interfaces

REACH_M = 2.0

c = sample_interfaces(500, 20250812, 30.0)
for k, v in c.items():
    x = redim_x(v.x.detach().cpu().numpy().ravel(), S)
    z = redim_x(v.z.detach().cpu().numpy().ravel(), S)

    d_floor = x                              # pit floor at x = 0
    d_top = np.abs(z - g.z_ground(x))        # cut face / bench / natural ground
    d_f1 = np.abs(g.x_f1(z) - x)             # F1 far-field Dirichlet
    d = np.minimum.reduce([d_floor, d_top, d_f1])

    print(f"{k:8s} min {d.min():7.2f} m   median {np.median(d):7.1f} m   "
          f"within {REACH_M} m: {(d < REACH_M).sum():3d}/{len(d)}")
    print()
print("marl thickness where the contact is sampled")
for k, v in c.items():
    x = redim_x(v.x.detach().cpu().numpy().ravel(), S)
    z = redim_x(v.z.detach().cpu().numpy().ravel(), S)
    thk = g.z_ground(x) - g.z_tm_top(x)
    print(f"{k:8s} x {x.min():6.1f}-{x.max():6.1f}   z {z.min():6.1f}-{z.max():6.1f}")
    print(f"         thickness  min {thk.min():6.2f}  median {np.median(thk):6.2f}  "
          f"max {thk.max():6.2f} m")