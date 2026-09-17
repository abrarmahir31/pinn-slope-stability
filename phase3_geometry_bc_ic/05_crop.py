"""Crop the Section 5-5' panel and upscale 3x for easier digitising."""
from PIL import Image

SRC = "figures/native_p5_x109.jpeg"
LEFT, TOP, RIGHT, BOTTOM = 15, 700, 1910, 1255
SCALE = 3

img = Image.open(SRC).convert("RGB")
crop = img.crop((LEFT, TOP, RIGHT, BOTTOM))
print(f"Cropped native: {crop.width} x {crop.height} px")

big = crop.resize((crop.width * SCALE, crop.height * SCALE),
                  Image.LANCZOS)
big.save("figures/section_5_5.png")
print(f"Upscaled {SCALE}x -> figures/section_5_5.png  "
      f"({big.width} x {big.height} px)")
