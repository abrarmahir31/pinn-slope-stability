"""Tests for `L_interface` (Step 3.2, item 2).

`interface_flux_jump` is already pinned by `tests/test_interface.py` (6 tests).
This module tests only how `L_interface` ASSEMBLES it: the two normalisations,
per-contact partitioning, and that gradients reach the network.

The anchor is the same field as `test_loss_bc.py`: psi = -(z - Z_WT) gives
uniform total head, so the Darcy flux is identically zero on BOTH sides of the
contact and the jump must vanish exactly -- across a 3000x K_s contrast, which
is the point. Finding 3 says that contrast reverses with wetness, so the
negative controls probe both a wet and a dry state.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nondim import SCALES as S
from loss import L_interface, _psi_field
from sampling import sample_interfaces, CONTACTS
from materials import load_materials
from step21_geometry.boundaries import Z_WT

N = 400
SEED = 20250812
MATS = load_materials()
DTYPE = torch.float64

ZW_STAR = Z_WT / S.L_ref
CONTACT_KEYS = tuple(CONTACTS)


@pytest.fixture(scope="module")
def ifaces():
    return sample_interfaces(N, SEED)


@pytest.fixture(scope="module")
def W_global(ifaces):
    return sum(float(c.w.sum()) for c in ifaces.values())


def call_L_interface(field, ifaces):
    return L_interface(field, ifaces, MATS, per_contact=True)


def PART(parts, key):
    return float(parts[f"interface_{key}"])


def hydrostatic(offset: float = 0.0, a: float = 0.0, b: float = 0.0):
    """Same anchor as test_loss_bc.py. a = b = 0 gives zero flux everywhere.

    Both gradient components are available because the contact follows
    z_tm_top(x), whose slope varies along its length -- a z-only field would
    probe the steep limbs strongly and the flat ones barely at all.
    """
    def field(x, z, t):
        dz = z - ZW_STAR
        psi = -dz / S.Pi_R_hz + offset + a * dz + b * x
        return psi + 0.0 * x + 0.0 * t
    return field


class TinyNet(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, 3),
        ).to(DTYPE)

    def forward(self, x, z, t):
        return self.net(torch.cat([x, z, t], dim=1))


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------
def test_returns_scalar_and_detached_per_contact_parts(ifaces):
    total, parts = call_L_interface(hydrostatic(a=1e-3), ifaces)

    assert total.ndim == 0
    assert "interface" in parts
    for key in CONTACT_KEYS:
        assert f"interface_{key}" in parts, f"interface_{key} missing"
    assert float(parts["interface"]) == pytest.approx(float(total.detach()),
                                                      rel=1e-12)
    for k, v in parts.items():
        assert not torch.as_tensor(v).requires_grad, f"parts[{k}] not detached"


def test_per_contact_parts_are_opt_in(ifaces):
    """`per_contact` defaults False, as `L_BC`'s `per_segment` does."""
    _, parts = L_interface(hydrostatic(a=1e-3), ifaces, MATS)
    assert "interface" in parts
    assert not any(k.startswith("interface_") for k in parts)


@pytest.mark.parametrize("a", [0.0, 1e-3, 1e-2])
def test_reweighted_parts_reconstruct_the_total(ifaces, W_global, a):
    """part * (w_contact / W_global), summed, == total. Pins both
    normalisations at once and documents that they are not additive."""
    total, parts = call_L_interface(hydrostatic(a=a), ifaces)
    reconstructed = sum(
        PART(parts, key) * float(coll.w.sum()) / W_global
        for key, coll in ifaces.items()
    )
    assert reconstructed == pytest.approx(float(total.detach()),
                                          rel=1e-9, abs=1e-30)


# ---------------------------------------------------------------------------
# The zero-flux anchor
# ---------------------------------------------------------------------------
def test_uniform_total_head_gives_no_jump(ifaces):
    """Zero flux on both sides, so no discontinuity -- across a 3000x K_s
    contrast. This is the assertion that a jump computed from the K ratio
    rather than from the fluxes would fail."""
    total, parts = call_L_interface(hydrostatic(), ifaces)
    assert float(total.detach()) < 1e-24
    for key in CONTACT_KEYS:
        assert PART(parts, key) < 1e-24, f"{key} has a jump at zero flux"


@pytest.mark.parametrize("key", CONTACT_KEYS)
def test_a_flowing_field_produces_a_jump(ifaces, key):
    """Negative control. Without it, `return 0.0` passes the test above.

    Measured against the contact's own anchor value rather than an absolute
    floor -- the two contacts sit at different depths and therefore different
    saturations, so their healthy magnitudes differ by orders.
    """
    _, anchor = call_L_interface(hydrostatic(), ifaces)
    _, flowing = call_L_interface(hydrostatic(a=1e-2, b=1e-2), ifaces)

    at_rest, moved = PART(anchor, key), PART(flowing, key)
    assert moved > 0.0, f"{key} is identically zero for every field"
    assert moved > 1e6 * max(at_rest, 1e-300), (
        f"{key} did not react to a flowing field: {at_rest:.3e} -> {moved:.3e}"
    )


def test_jump_is_symmetric_in_the_normal_sign(ifaces):
    """`interface_flux_jump` squares the normal, so flipping it must not change
    the loss. Guards the sampler's orientation choice from becoming load-
    bearing without anyone noticing."""
    import dataclasses

    flipped = {k: dataclasses.replace(v, nx=-v.nx, nz=-v.nz)
               for k, v in ifaces.items()}
    a, _ = call_L_interface(hydrostatic(a=1e-2), ifaces)
    b, _ = call_L_interface(hydrostatic(a=1e-2), flipped)
    assert float(b.detach()) == pytest.approx(float(a.detach()), rel=1e-12)


# ---------------------------------------------------------------------------
# Both contacts are really being evaluated
# ---------------------------------------------------------------------------
def test_each_contact_uses_its_own_upper_material(ifaces):
    """While Mk and Mk_d share K_s, alpha and n the two contacts should give
    the same jump for the same field state. This test does NOT assert they are
    equal -- that would bake finding 2 in. It asserts the loss reads mat_a from
    the key, by checking a deliberately mismatched call differs."""
    field = hydrostatic(a=1e-2)
    _, parts = call_L_interface(field, ifaces)

    swapped = {"mk_tm": ifaces["mkd_tm"], "mkd_tm": ifaces["mk_tm"]}
    _, swapped_parts = call_L_interface(field, swapped)

    assert PART(parts, "mk_tm") != pytest.approx(PART(swapped_parts, "mk_tm"),
                                                 rel=1e-9, abs=1e-40), (
        "swapping the contact sets changed nothing; mat_a may be hardcoded"
    )


def test_dropping_a_contact_lowers_the_total(ifaces):
    """The total accumulates across contacts. If it silently used only the
    first key, removing the second would not change it."""
    full, _ = call_L_interface(hydrostatic(a=1e-2), ifaces)
    one = {"mk_tm": ifaces["mk_tm"]}
    partial, _ = call_L_interface(hydrostatic(a=1e-2), one)
    assert float(partial.detach()) != pytest.approx(float(full.detach()),
                                                    rel=1e-9, abs=1e-40)



# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------
def test_gradients_reach_the_network(ifaces):
    net = TinyNet()
    total, _ = call_L_interface(_psi_field(net), ifaces)
    total.backward()

    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads), "a parameter received no gradient"
    assert max(float(g.abs().max()) for g in grads) > 0.0, "all gradients zero"


def test_weight_is_order_one_against_L_PDE(ifaces):
    """w_interface = 1.0 is only sane while the terms are the same order. If
    this drifts, the starting weight needs revisiting before GradNorm."""
    net = TinyNet()
    total, _ = call_L_interface(_psi_field(net), ifaces)
    assert 1e-8 < float(total.detach()) < 1e2, (
        f"interface loss at {float(total.detach()):.3e} is far from the ~3e-3 "
        f"scale L_PDE sits at; w_interface = 1.0 may no longer be balanced"
    )