"""tests/test_residuals.py

Covers residuals.py paths that test_richards_hydrostatic.py does not.
"""

import torch

from src.config import tiny, BOUNDS
from src.model import PINN
from src.sampling import sample_interior
from src.loss import psi_of
from src.residuals import richards_residual
from src.materials import load_materials


def test_richards_accepts_material_not_swcc():
    """swcc_from_material's branch was dead until L_PDE hit it: `_m.C`
    did not exist and no test ever ran that path."""
    mats = load_materials()
    c = sample_interior(60)
    net = PINN(tiny(), BOUNDS)

    def fields(x, z, t):
        return psi_of(net(x, z, t))

    R = richards_residual(fields, c.x, c.z, c.t, mats["Tm"])
    assert torch.isfinite(R).all()
    assert R.shape == (len(c), 1)