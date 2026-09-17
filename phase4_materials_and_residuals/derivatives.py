"""
src/derivatives.py — the automatic-differentiation layer.
=========================================================

Nothing in this file knows any physics. It knows only how to differentiate a
tensor with respect to a leaf tensor and how to fail loudly when that is not
possible. Every derivative taken anywhere in the project goes through `grad`,
so `create_graph=True` is set in exactly one place and cannot be forgotten.

Three behaviours here are deliberate and are the reason this is a module rather
than a three-line lambda:

1.  `create_graph=True` by default. Without it the returned gradient is a leaf
    and the *second* derivative silently comes back as None. The Richards
    residual needs second derivatives, so this default is load-bearing.

2.  Unused inputs return zeros, not None. An analytic test field such as
    psi = -(z - 197) does not depend on x or t at all; autograd reports that
    by returning None. Physically dpsi/dt IS zero there, so zeros is the
    correct answer and lets analytic fields be passed to the same residual
    code as the network. See `strict=True` to turn this off when you want the
    stricter behaviour (e.g. checking a network really is wired to all inputs).

3.  Constants are tolerated. A field that is identically zero (u = v = 0 in a
    seepage-only test) has no grad_fn, and autograd raises. Returning zeros is
    again the physically correct answer.

Convention: all field tensors are shape (N, 1) or (N,) and match the shape of
the coordinate they are differentiated against.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

__all__ = [
    "as_inputs",
    "grad",
    "grads",
    "laplacian",
    "divergence",
]


def as_inputs(*arrays, dtype: torch.dtype = torch.float64,
              device: str | torch.device = "cpu") -> list[Tensor]:
    """Turn coordinate arrays into differentiable LEAF tensors.

    Leaf-ness matters: `torch.autograd.grad(psi, x)` only resolves if `x` is a
    leaf that participated in the graph. Anything built by slicing or
    concatenating outside the network is not a leaf. Hence detach().clone().

    float64 is the default on purpose. In float32 the hydrostatic Richards test
    bottoms out around 1e-7 instead of 1e-14, which is not tight enough to
    distinguish "exactly zero" from "nearly cancelled".
    """
    out: list[Tensor] = []
    for a in arrays:
        t = a if isinstance(a, Tensor) else torch.as_tensor(a)
        t = t.to(dtype=dtype, device=device).detach().clone()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        out.append(t.requires_grad_(True))
    return out


def grad(y: Tensor, x: Tensor, *, create_graph: bool = True,
         strict: bool = False) -> Tensor:
    """d y / d x, elementwise, with the graph kept alive.

    Parameters
    ----------
    y, x : tensors of the same shape. `x` must be a leaf with requires_grad.
    create_graph : keep the graph so the result can be differentiated again.
        Only set False for a final diagnostic you will not differentiate.
    strict : raise instead of returning zeros when y does not depend on x.
    """
    if not y.requires_grad:
        if strict:
            raise RuntimeError(
                "grad(): y has no grad_fn — it is a constant or was detached. "
                "Check for an accidental .item(), .detach() or torch.no_grad()."
            )
        return torch.zeros_like(x)

    g = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=not strict,
    )[0]

    if g is None:
        if strict:
            raise RuntimeError(
                "grad(): y does not depend on x (autograd returned None). "
                "If that is expected, call with strict=False."
            )
        return torch.zeros_like(x)
    return g


def grads(y: Tensor, xs: Sequence[Tensor], **kw) -> list[Tensor]:
    """[dy/dx for x in xs]. Convenience only."""
    return [grad(y, x, **kw) for x in xs]


def laplacian(y: Tensor, xs: Sequence[Tensor], **kw) -> Tensor:
    """sum_i d2y/dxi2. Note this is the CONSTANT-coefficient Laplacian.

    Do NOT use it for div[K(psi) grad psi] — that needs the divergence form
    below, or the product rule written out by hand. Using this by mistake is
    the single easiest way to silently drop the dK/dpsi |grad psi|^2 term.
    """
    total = None
    for x in xs:
        d2 = grad(grad(y, x, **kw), x, **kw)
        total = d2 if total is None else total + d2
    return total


def divergence(components: Sequence[Tensor], xs: Sequence[Tensor], **kw) -> Tensor:
    """div F = sum_i dF_i/dx_i, with F_i already assembled (e.g. K * dpsi/dxi).

    Taking the divergence of the assembled flux — rather than expanding the
    product rule by hand — means autograd computes dK/dpsi * dpsi/dxi for you.
    That chain rule is where sign errors live, so we let the machine do it.
    """
    if len(components) != len(xs):
        raise ValueError(f"divergence(): {len(components)} components vs "
                         f"{len(xs)} coordinates")
    total = None
    for f, x in zip(components, xs):
        d = grad(f, x, **kw)
        total = d if total is None else total + d
    return total
