"""Confirm the PDF is reachable and every needed package imports."""
from pathlib import Path

PDF = Path("/mnt/d/Books/Thesis/My thesis/j.enggeo.2014.08.005.pdf")

print(f"PDF exists : {PDF.exists()}")
if PDF.exists():
    print(f"Size       : {PDF.stat().st_size} bytes   (expect 7299963)")
else:
    print("  -> run:  ls ../*.pdf   and correct the path above")

print()
for mod in ["fitz", "PIL", "numpy", "scipy", "pandas", "matplotlib"]:
    try:
        __import__(mod)
        print(f"  ok      {mod}")
    except ImportError as e:
        print(f"  MISSING {mod}   ({e})")
