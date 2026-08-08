"""Loss terms for the coupled hydro-mechanical PINN.

PINN.forward(x, z, t) takes three separate (N, 1) leaf tensors and
returns a stacked (N, 3) tensor with columns [psi*, u*, v*]. The slicing
helpers below exist because residuals._psi_of mishandles that shape (it
takes the isinstance(Tensor) branch and returns all three columns).
Slicing lives here rather than in residuals.py to keep the
mutation-tested Richards path untouched.
"""

from __future__ import annotations

import torch

from src.nondim import SCALES, Scales
from src.sampling import load_initial


def psi_of(out: torch.Tensor) -> torch.Tensor:
    """Pressure head column, shape (N, 1)."""
    return out[:, 0:1]


def uv_of(out: torch.Tensor) -> torch.Tensor:
    """Displacement columns (u*, v*), shape (N, 2)."""
    return out[:, 1:3]


def ic_targets(path: str | None = None, s: Scales = SCALES):
    """(Collocation at t*=0, psi0_star) with psi0 non-dimensionalised.

    load_initial returns `targets` as a raw passthrough of the cache --
    coordinates are divided by L_ref inside it, but psi0 is NOT. It is
    stored in metres, so it must be divided by H_ref here to be
    comparable with the network's psi* output. Omitting this is a
    factor-30 error.
    """
    coll, targets = load_initial() if path is None else load_initial(path)

    psi0 = torch.as_tensor(
        targets["psi0"], dtype=coll.x.dtype, device=coll.x.device
    ).reshape(-1, 1)

    return coll, (psi0 / s.H_ref).detach()

def L_IC(net, coll, psi0_star):
    """mean_w[(psi* - psi0*)^2] + mean_w[u*^2 + v*^2] at t* = 0.

    Displacements are homogeneous at t = 0, so the mechanical term needs
    no target data. sig_v/sig_h in the cache are the sigma_0 lookup for
    the mechanical residual, not IC targets, so this term does not wait
    on the gravity warm-up.

    `coll.w` is already mean-normalised in load_initial, so dividing by
    its sum gives a true weighted mean comparable to an unweighted MSE.
    """
    out = net(coll.x, coll.z, coll.t)

    w = coll.w.detach()
    wsum = w.sum()

    r_psi = psi_of(out) - psi0_star
    head = (w * r_psi.pow(2)).sum() / wsum

    uv = uv_of(out)
    disp = (w * uv.pow(2).sum(dim=1, keepdim=True)).sum() / wsum

    return head + disp, {"ic_head": head.detach(), "ic_disp": disp.detach()}
