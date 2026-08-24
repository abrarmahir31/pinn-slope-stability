"""Day-2 lab-PC GPU sanity check.

Run inside the activated env:
    python tools/gpu_check.py
"""
import sys

import torch

print(f"torch               {torch.__version__}")
print(f"built for CUDA      {torch.version.cuda}")
print(f"cuda.is_available   {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\nFAIL: CPU-only build or driver problem. Stop here.")
    sys.exit(1)

props = torch.cuda.get_device_properties(0)
cap = torch.cuda.get_device_capability(0)
print(f"device              {props.name}")
print(f"capability          sm_{cap[0]}{cap[1]}")
print(f"VRAM                {props.total_memory / 1e9:.1f} GB")

a = torch.randn(512, 512, device="cuda")
b = torch.randn(512, 512, device="cuda")
c = a @ b
torch.cuda.synchronize()
print(f"\nmatmul              ok  {tuple(c.shape)}  mean={c.mean().item():+.4f}")

x = torch.linspace(0.0, 1.0, 128, device="cuda").unsqueeze(-1).requires_grad_(True)
y = x ** 3
dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
d2y = torch.autograd.grad(dy, x, torch.ones_like(dy), create_graph=True)[0]

err1 = (dy - 3.0 * x ** 2).abs().max().item()
err2 = (d2y - 6.0 * x).abs().max().item()
print(f"dy/dx   max err     {err1:.3e}")
print(f"d2y/dx2 max err     {err2:.3e}   (expect < 1e-4 in float32)")

if err2 > 1e-4:
    print("\nFAIL: second-order autograd is wrong. Do not proceed to training.")
    sys.exit(1)

print("\nRecord this triple in your notebook:")
print(f"  torch {torch.__version__} | built for CUDA {torch.version.cuda} | {props.name}")
print("\nAll checks passed.")
