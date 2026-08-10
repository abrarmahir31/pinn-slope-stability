# tests/test_bounds.py
def test_bounds_literal_matches_geometry():
    derived = bounds_from_geometry()
    for k in BOUNDS:
        assert abs(BOUNDS[k] - derived[k]) < 1e-3, k