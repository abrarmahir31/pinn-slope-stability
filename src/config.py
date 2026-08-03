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
    