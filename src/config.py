from src.nondim import SCALES
from dataclasses import dataclass

@dataclass
class NetConfig:
    n_inputs:   int = 3
    n_outputs:  int = 3
    n_layers:   int = 8
    n_neurons:  int = 64
    activation: str = "tanh"
    init:       str = "xavier"
    dtype: str = "float64"
    device:     str = "cuda"
    seed:       int = 42
    fourier_features: bool = False
    hard_bc:          bool = False
    

# Dimensionless domain bounds (L_ref = 170 m, T_ref = 86400 s)
# TODO: source these from geometry.py once Z_BASE and X_MK_DIVIDE are settled
BOUNDS = {
    "x_min": -0.0022114108801415335,  "x_max": 1.577,   # -0.38 .. 268.05 m / 170
    "z_min": 1.173,  "z_max": 2.106,   # 199.47-357.97 m / 170
    "t_min": 0.0,    "t_max": 30.0,    # 30 days
}


def tiny() -> NetConfig:
    """Laptop config: small enough for CPU, float64 for clean gradient checks."""
    return NetConfig(n_layers=3, n_neurons=20, dtype="float64", device="cpu")


def full() -> NetConfig:
    """Lab PC config: the architecture of record, ~29.6k parameters."""
    return NetConfig()


def resolve_device(requested: str) -> str:
    """Fall back to CPU when CUDA is unavailable (laptop vs lab PC)."""
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def bounds_from_geometry(t_max=30.0, s=SCALES) -> dict:
    from src.sampling import domain_bbox
    x0, x1, z0, z1 = domain_bbox()
    return {"x_min": x0/s.L_ref, "x_max": x1/s.L_ref,
            "z_min": z0/s.L_ref, "z_max": z1/s.L_ref,
            "t_min": 0.0, "t_max": t_max}