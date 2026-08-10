from src.config import BOUNDS, bounds_from_geometry, tiny
from src.model import PINN
from src.sampling import sample_interior


def test_bounds_literal_matches_geometry():
    derived = bounds_from_geometry()
    for k in BOUNDS:
        assert abs(BOUNDS[k] - derived[k]) < 1e-3, k


def test_sampled_points_scale_into_unit_box():
    net = PINN(tiny(), BOUNDS)
    c = sample_interior(2000)
    for name, v, lo, hi in [("x", c.x, net.x_min, net.x_max),
                            ("z", c.z, net.z_min, net.z_max),
                            ("t", c.t, net.t_min, net.t_max)]:
        s = PINN._scale(v, lo, hi)
        assert s.min() >= -1.01 and s.max() <= 1.01, \
            f"{name}: {s.min():.4f}..{s.max():.4f}"