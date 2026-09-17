"""Crop Fig. 19d (Section 5 slope profiles) and upscale 4x for digitising."""
from PIL import Image

SRC = "figures/native_p18_x407.jpeg"
LEFT, TOP, RIGHT, BOTTOM = 0, 1600, 994, 2245
SCALE = 4

img = Image.open(SRC).convert("RGB")
crop = img.crop((LEFT, TOP, RIGHT, BOTTOM))
print(f"Cropped native: {crop.width} x {crop.height} px")

big = crop.resize((crop.width * SCALE, crop.height * SCALE), Image.LANCZOS)
big.save("figures/fig19d_section5.png")
print(f"Upscaled {SCALE}x -> figures/fig19d_section5.png ({big.width} x {big.height} px)")
