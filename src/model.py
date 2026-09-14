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

class NearPhysical(nn.Module):
    """PINN offset to start at the physical state (the 6cbd402 init).

        psi* = psi_0* + eps_psi * net,   u*, v* = eps_uv * net

    psi_0* is the Step 2.3 IC, psi = -(z - z_wt), z_wt = 197.0 m a.s.l.,
    non-dimensionalised by H_ref. Without it the raw net emits psi* ~ 1,
    i.e. 165 m of suction, where van Genuchten K_r is numerically zero and
    every flux term collapses twelve orders -- the regime that fooled the
    Step 3.3b tests. The Step 3.4 weights are only valid for this init.

    Wraps rather than subclasses so PINN keeps its "contains no physics"
    property: the IC is Step 2.3's, and it lives here.

    THE SATURATION PROBLEM (Day 27)
    ------------------------------
    The output layer is a bare `nn.Linear`, so `raw` is UNBOUNDED. With
    psi_0* in [-0.976, -0.018], a point at the domain floor needs only
    raw > 0.06 at eps_psi = 0.3 to cross psi = 0. Measured on an untrained
    8x64 net over three seeds, uniform draw, N = 4000:

        eps_psi = 3e-3   0.0% of points at psi >= 0, psi_max -1.5 .. -2.9 m
        eps_psi = 0.3    19.7% / 35.0% / 15.1%,      psi_max +54 .. +185 m
        prod02 @ 2000    0.0%,                       psi_max -3.1 m

    Section 5 is unsaturated everywhere by construction -- Z_WT = 197 m is 3 m
    BELOW Z_BASE = 200 m -- so a sixth to a third of the domain initialises in
    a state the solution never occupies, at up to 185 m of positive head. Van
    Genuchten K_r with n = 1.2 is near-singular there, and that is what makes
    `w^[pde_richards]` move 1264x across (seed, draw) pairs: 2-3 points out of
    4000 carry 90% of L_PDE_richards, all of them at psi ~ 0. Training removes
    it by step 2000, but the Step 3.4 seed is measured at step 0.

    `mode` selects how psi is held unsaturated. **The default is unchanged**
    (`"unbounded"`, the shipped behaviour) because the alternatives trade IC
    fidelity against seed measurability and NOTHING HAS BEEN DECIDED -- the
    evidence so far is t = 0 gradient norms on one seed triple, which cannot
    separate them. See `scripts/ansatz_ablation.py` and DECISIONS.md D-A.1.

        "unbounded"  psi* = psi0* + eps*raw                  (current)
        "cap"        psi* = -softplus(-k(psi0* + eps*raw))/k  smooth cap at 0
        "exp"        psi* = psi0* * exp(eps*raw)              multiplicative

    IC distortion at psi_0 = -3 m (raw = 0), and w^[pde_richards] spread at
    N = 1200, 8x64, anchor bc_mech, over three (net, draw) seed pairs:

        mode / cap_k        IC err     spread      w^ range
        unbounded             0.0%    1264.1x      3.4e-3 .. 11.4
        cap, k = 20         145.1%        4.1x     17.9 .. 146
        cap, k = 100          8.3%       18.4x     0.19 .. 5.5
        cap, k = 1000         0.0%      100.0x     2.0e-3 .. 0.26
        exp                   0.0%        3.1x     1.4e4 .. 4.9e4

    The trade-off is monotone: the spread falls as the cap softens, and
    softening is exactly what distorts the IC in the bottom few metres, which
    is the ONLY band where the Richards physics is live (see
    scripts/inert_fraction.py). k = 20 is unusable for that reason and k = 1000
    does not fix the spread. `exp` costs nothing on the IC and holds psi away
    from saturation, but it can barely wet (psi_max -1.3 .. -3.1 m at init),
    which walks back toward the Day 25 regime that raising eps_psi existed to
    escape, and it made pde_mech's spread worse (52.5x).

    CONDITIONAL VALIDITY. psi <= 0 is guaranteed only by the current model
    configuration: `INFILTRATION_MODE = "capacity_limited"` caps the rain flux
    at K_s so the surface approaches saturation asymptotically rather than
    ponding, and `SEEPAGE_FACE_MODE = "noflow"` zeroes the cut face. Both are
    open items. If either changes, a bounded ansatz becomes a modelling
    assumption rather than a restatement of the geometry, and must be declared
    as one.
    """
    Z_WT = 197.0
    MODES = ("unbounded", "cap", "exp")

    def __init__(self, pinn, s=None, eps_psi=0.3, eps_uv=1e-3,
                 mode="unbounded", cap_k=100.0):
        super().__init__()
        from src.nondim import SCALES
        if mode not in self.MODES:
            raise ValueError(f"mode={mode!r} not in {self.MODES}")
        if mode == "cap" and cap_k <= 0:
            raise ValueError(f"cap_k must be positive, got {cap_k}")
        self.pinn = pinn
        self.s = s or SCALES
        self.eps_psi, self.eps_uv = eps_psi, eps_uv
        self.mode, self.cap_k = mode, float(cap_k)

    @property
    def device(self):
        return self.pinn.device

    def summary(self):
        tail = "" if self.mode == "unbounded" else (
            f", mode={self.mode}" + (f" k={self.cap_k:g}"
                                     if self.mode == "cap" else ""))
        return (f"{self.pinn.summary()} | near-physical init "
                f"(eps_psi={self.eps_psi:g}, eps_uv={self.eps_uv:g}{tail})")

    def psi_star(self, z, raw_psi):
        """psi* from the raw network channel. Separated so tests and
        `scripts/ansatz_ablation.py` can probe the map without a forward pass,
        and so the three branches sit in one place rather than in `forward`."""
        psi0 = -(z * self.s.L_ref - self.Z_WT) / self.s.H_ref
        if self.mode == "unbounded":
            return psi0 + self.eps_psi * raw_psi
        if self.mode == "cap":
            k = self.cap_k
            # softplus is stable for large |k*p|: at p << 0, -k*p is large and
            # positive, softplus(-k*p) -> -k*p, so the result -> p untouched.
            return -nn.functional.softplus(-k * (psi0 + self.eps_psi * raw_psi)) / k
        # "exp": psi0 < 0 everywhere in Section 5, so the product is strictly
        # negative for any finite raw. Saturation is approached, never crossed.
        return psi0 * torch.exp(self.eps_psi * raw_psi)

    def forward(self, x, z, t):
        raw = self.pinn(x, z, t)
        return torch.cat([self.psi_star(z, raw[:, 0:1]),
                          self.eps_uv * raw[:, 1:3]], dim=1)