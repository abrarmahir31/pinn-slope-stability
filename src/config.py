from dataclasses import dataclass

@dataclass
class NetConfig:
    n_inputs:   int = 3
    n_outputs:  int = 3
    n_layers:   int = 8
    n_neurons:  int = 64
    activation: str = "tanh"
    init:       str = "xavier"
    dtype:      str = "float32"
    device:     str = "cuda"
    seed:       int = 42
    fourier_features: bool = False
    hard_bc:          bool = False
    

# Dimensionless domain bounds (L_ref = 170 m, T_ref = 86400 s)
# TODO: source these from geometry.py once Z_BASE and X_MK_DIVIDE are settled
BOUNDS = {
    "x_min": 0.0, "x_max": 4.41,   # 750 m / 170
    "z_min": 0.0, "z_max": 1.18,   # 200 m / 170
    "t_min": 0.0, "t_max": 30.0,   # 30 days
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
