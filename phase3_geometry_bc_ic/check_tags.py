import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import geometry as g

for x in [50, 85, 95, 150, 200, 260]:
    for z in [280, 300, 320, 340]:
        print(f"x={x:4d} z={z:4d} -> {g.material_tag(x, z)}")
    print()
