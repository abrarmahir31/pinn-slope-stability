"""Where does rain actually enter? Run before writing L_BC.

Prints, per rainfall segment, the material split and the fraction of the
prescribed rainfall that the capped BC admits.
"""
import numpy as np
import geometry as g
import boundaries as b

segs = b.sample_boundaries(n=4000, seed=0)

print(f"rain = {b.RAIN_FLUX:.3e} m/s = {b.RAIN_FLUX * 3.6e6:.1f} mm/hr\n")
print(f"{'segment':<18}{'material':<8}{'pts':>7}{'share':>9}"
      f"{'q_applied/q_rain':>20}")
print("-" * 62)

total_rain = total_applied = 0.0
for name in ("natural_ground", "bench"):
    pts = segs[name]
    nrm = b.outward_normal(g.z_ground, pts[:, 0])
    q_rain = b.RAIN_FLUX * nrm[:, 1]
    cap = b.infiltration_capacity(pts[:, 0], pts[:, 1])
    q_app = np.minimum(q_rain, cap)
    tag = g.material_tag(pts[:, 0], pts[:, 1])

    total_rain += q_rain.sum()
    total_applied += q_app.sum()

    for m in ("Mk", "Mk_d", "Tm"):
        sel = tag == m
        if not sel.any():
            continue
        print(f"{name:<18}{m:<8}{sel.sum():>7}{sel.mean():>8.1%}"
              f"{q_app[sel].sum() / q_rain[sel].sum():>20.4%}")

print("-" * 62)
print(f"{'TOTAL admitted':<33}{total_applied / total_rain:>28.2%}")
print("\nIf this is a few per cent, the triggered case is a Tm-infiltration\n"
      "problem and the marl behaves as a near-impermeable lid. That is a\n"
      "results-chapter statement, not a boundary-condition detail.")