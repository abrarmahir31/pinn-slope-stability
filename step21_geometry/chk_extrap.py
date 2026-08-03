import numpy as np, geometry as g
for lo, hi, lbl in [(g.X_MIN, g.X_MAX, "full domain 0-268"),
                    (3.8, 266.1, "traced range only")]:
    x = np.linspace(lo, hi, 4000)
    t = g.z_ground(x) - g.z_tm_top(x)
    i = int(np.argmin(t))
    print(f"{lbl:22s} min t = {t.min():+.3f} m at x = {x[i]:.1f}")
