import pytest
import torch

from src.config import NetConfig, BOUNDS, tiny, full
from src.model import PINN


def make(cfg):
    return PINN(cfg, BOUNDS)


def sample(n, cfg):
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    kw = dict(dtype=dtype, device=cfg.device, requires_grad=True)
    return (torch.rand(n, 1, **kw),
            torch.rand(n, 1, **kw),
            torch.rand(n, 1, **kw))


def test_parameter_count():
    """8x64 with 3 in, 3 out: 256 + 7*4160 + 195 = 29571."""
    assert make(full()).n_parameters() == 29_571


def test_output_shape():
    cfg = tiny()
    out = make(cfg)(*sample(17, cfg))
    assert out.shape == (17, 3)


def test_second_derivatives_exist():
    """The critical one: Richards needs d2psi/dx2, which needs create_graph."""
    cfg = tiny()
    net = make(cfg)
    x, z, t = sample(10, cfg)
    psi = net(x, z, t)[:, 0:1]

    dpsi_dx = torch.autograd.grad(
        psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    assert dpsi_dx is not None

    d2psi_dx2 = torch.autograd.grad(
        dpsi_dx, x, grad_outputs=torch.ones_like(dpsi_dx), create_graph=True)[0]
    assert d2psi_dx2 is not None
    assert torch.isfinite(d2psi_dx2).all()


def test_normalisation_hits_corners():
    cfg = tiny()
    net = make(cfg)
    dtype = torch.float64
    lo = torch.tensor([[BOUNDS["x_min"]]], dtype=dtype)
    hi = torch.tensor([[BOUNDS["x_max"]]], dtype=dtype)
    assert torch.allclose(net._scale(lo, net.x_min, net.x_max),
                          torch.tensor([[-1.0]], dtype=dtype))
    assert torch.allclose(net._scale(hi, net.x_min, net.x_max),
                          torch.tensor([[1.0]], dtype=dtype))


def test_determinism():
    cfg = tiny()
    a, b = make(cfg), make(cfg)
    inputs = sample(5, cfg)
    assert torch.allclose(a(*inputs), b(*inputs))


def test_rejects_relu():
    cfg = tiny()
    cfg.activation = "relu"
    with pytest.raises(ValueError):
        make(cfg)
