"""Fully-connected feed-forward PINN for coupled hydro-mechanical slope stability.

Inputs  : (x*, z*, t*)  dimensionless coordinates
Outputs : (psi*, u*, v*) dimensionless pressure head and displacements

Contains no physics. All PDE residuals live in the loss module (Step 3.2).
"""
from src.config import resolve_device
import torch
import torch.nn as nn

ACTIVATIONS = {"tanh": torch.tanh}
DTYPES = {"float32": torch.float32, "float64": torch.float64}


class PINN(nn.Module):
    def __init__(self, cfg, bounds):
        super().__init__()
        torch.manual_seed(cfg.seed)

        self.cfg = cfg
        dtype = DTYPES[cfg.dtype]

        for key in ("x_min", "x_max", "z_min", "z_max", "t_min", "t_max"):
            self.register_buffer(key, torch.tensor(bounds[key], dtype=dtype))

        dims = ([cfg.n_inputs]
                + [cfg.n_neurons] * cfg.n_layers
                + [cfg.n_outputs])
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

        try:
            self.act = ACTIVATIONS[cfg.activation]
        except KeyError:
            raise ValueError(
                f"activation {cfg.activation!r} not supported; PINNs need a "
                "C-infinity activation for second derivatives"
            )

        self._init_weights()
        self.device = resolve_device(cfg.device)
        self.to(dtype=dtype, device=self.device)

    def _init_weights(self):
        # Xavier with the tanh gain (5/3). PyTorch's default is Kaiming-uniform,
        # tuned for ReLU, which under-scales variance through a deep tanh stack.
        gain = nn.init.calculate_gain(self.cfg.activation)
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _scale(v, lo, hi):
        # Map to [-1, 1]. Out-of-place only: in-place ops on leaf tensors
        # break autograd.
        return 2.0 * (v - lo) / (hi - lo) - 1.0

    def forward(self, x, z, t):
        """x, z, t: (N, 1) tensors, each requires_grad=True for the residuals.

        They are concatenated INSIDE forward so each stays a distinct leaf and
        torch.autograd.grad(psi, x) resolves correctly in Step 3.3.
        """
        xn = self._scale(x, self.x_min, self.x_max)
        zn = self._scale(z, self.z_min, self.z_max)
        tn = self._scale(t, self.t_min, self.t_max)

        h = torch.cat([xn, zn, tn], dim=1)
        for layer in self.layers[:-1]:
            h = self.act(layer(h))
        return self.layers[-1](h)          # (N, 3) -> psi*, u*, v*

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        return (f"PINN {self.cfg.n_layers}x{self.cfg.n_neurons} "
                f"{self.cfg.activation} | {self.n_parameters():,} params | "
                f"{self.cfg.dtype} on {self.device}")
