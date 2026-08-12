"""Path setup for the test suite.

No packaging (no pyproject/setup.py), so pytest puts only tests/ on sys.path.
Both src/ and src/step21_geometry/ are needed: boundaries.py does a flat
`import geometry as g`, so the inner directory has to be importable too.

Living here rather than in each test file means every module resolves the same
`nondim` -- a per-file path hack lets a new test import a different one and
nothing tells you.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
for p in (SRC / "step21_geometry", SRC):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
