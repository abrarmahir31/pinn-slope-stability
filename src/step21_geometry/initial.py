"""
Initial conditions for Section 5, Isikdere pit (Step 2.3).

Z_WT = 197.0 m a.s.l. -- karstic bedrock aquifer, Ulusay et al. (2014) Sec 5.3.
Chosen as the measured MAXIMUM of the 113-197 m range because:
  1. Section 5 is on the northern pit wall; regional flow is N -> SSW, so
     heads are highest at the northern end of the range.
  2. A higher table means less suction and a LOWER F -- conservative.
  3. It matches the F1 far-field Dirichlet datum set in boundaries.py.
  4. Consistent with "head at least 25 m below the final pit bottom" (Sec 5.2),
     which is a ceiling, not a value, and is not used to derive Z_WT.

Z_WT = 197.0 < Z_BASE = 200.0, so the domain is entirely unsaturated at t = 0
and the saturated branch below is provably empty for this section. It is coded
anyway: Section 7 needs it, and an empty branch that is asserted empty is a
check, not dead weight.
"""

import numpy as np
try:
    from boundaries import Z_WT          # single source of truth
except ModuleNotFoundError:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from src.step21_geometry.boundaries import Z_WT


def psi_initial(x, z):
    """t = 0 pressure head, hydrostatic about the karstic water table.

    psi > 0  saturated      (z < Z_WT)  -- empty for Section 5
    psi < 0  matric suction (z > Z_WT)

    One expression covers both branches; an explicit conditional here is
    only a way to introduce a sign error.
    """
    return Z_WT - np.asarray(z, dtype=float)


if __name__ == "__main__":
    for z, want in [(197.0, 0.0), (200.0, -3.0), (271.13, -74.13),
                    (332.70, -135.70), (364.87, -167.87)]:
        got = float(psi_initial(0.0, z))
        flag = "ok" if abs(got - want) < 5e-3 else "MISMATCH"
        print(f"z = {z:8.2f}  psi0 = {got:9.3f}  expected {want:9.3f}  {flag}")
