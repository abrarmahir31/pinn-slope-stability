"""Verify the environment. Run inside the activated env:  python tools/verify_env.py"""
import importlib
import sys

print(f"Python  {sys.version.split()[0]}  ({sys.executable})\n")

ok = True
for name in ("numpy", "scipy", "matplotlib", "pandas", "torch"):
    try:
        m = importlib.import_module(name)
        print(f"  [ok]   {name:12s} {getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        ok = False
        print(f"  [FAIL] {name:12s} {e}")

try:
    import torch
    print(f"\n  torch built for CUDA : {torch.version.cuda}")
    print(f"  cuda.is_available()  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device               : {torch.cuda.get_device_name(0)}")
    else:
        print("  device               : CPU only (expected on the laptop)")
except Exception:
    pass

sys.exit(0 if ok else 1)
