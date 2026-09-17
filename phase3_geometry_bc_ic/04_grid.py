"""Grid overlay on the native extract so crop bounds can be read off."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SRC = "figures/native_p5_x109.jpeg"      # <-- EDIT if a different file is Fig. 5

img = Image.open(SRC)
print(f"Native: {img.width} x {img.height} px")

fig, ax = plt.subplots(figsize=(img.width / 200, img.height / 200), dpi=200)
ax.imshow(img)
step = 100
ax.set_xticks(range(0, img.width, step))
ax.set_yticks(range(0, img.height, step))
ax.grid(color="red", alpha=0.5, lw=0.4)
ax.tick_params(labelsize=4)
plt.setp(ax.get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig("figures/grid_p5.png", dpi=200)
print("written: figures/grid_p5.png")
