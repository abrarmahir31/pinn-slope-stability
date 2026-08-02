"""Verify the environment: imports each package and prints its version.

Run inside the activated env:
    python verify_env.py
"""
import importlib
import sys

print(f"Python  {sys.version.split()[0]}  ({sys.executable})\n")

checks = [
    ("numpy",      "numpy"),
    ("scipy",      "scipy"),
    ("matplotlib", "matplotlib"),
    ("torch",      "torch"),
]

ok = True
for label, module in checks:
    try:
        m = importlib.import_module(module)
        print(f"  [ok]   {label:12s} {getattr(m, '__version__', 'unknown')}")
    except Exception as e:                      # noqa: BLE001
        ok = False
        print(f"  [FAIL] {label:12s} {e}")

# PyRosetta is optional / installed separately
try:
    import pyrosetta                            # noqa: F401
    pyrosetta.init(silent=True)
    print(f"  [ok]   {'pyrosetta':12s} {pyrosetta.__version__}")
except Exception as e:                          # noqa: BLE001
    print(f"  [--]   {'pyrosetta':12s} not installed / not licensed ({type(e).__name__})")

# Confirm PyTorch really is the CPU build
try:
    import torch
    print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}  (expected: False for CPU build)")
except Exception:
    pass

sys.exit(0 if ok else 1)
