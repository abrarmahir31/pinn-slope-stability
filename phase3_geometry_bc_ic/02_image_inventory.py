"""Inventory embedded images on candidate pages: native size and effective DPI."""
import fitz

PDF = "/mnt/d/Books/Thesis/My thesis/j.enggeo.2014.08.005.pdf"
PAGES = range(2, 10)          # candidate pages from 01_find_figure

doc = fitz.open(PDF)
print(f"{'idx':>4} {'xref':>6} {'native px':>15} {'placed pt':>15} {'eff. DPI':>9}  cs")
print("-" * 70)

for i in PAGES:
    page = doc[i]
    for img in page.get_images(full=True):
        xref = img[0]
        info = doc.extract_image(xref)
        w, h = info["width"], info["height"]
        try:
            rect = page.get_image_rects(xref)[0]
            dpi_x = w / (rect.width / 72.0)
            dpi_y = h / (rect.height / 72.0)
            placed = f"{rect.width:.0f}x{rect.height:.0f}"
            dpi = f"{dpi_x:.0f}/{dpi_y:.0f}"
        except Exception:
            placed, dpi = "?", "?"
        print(f"{i:>4} {xref:>6} {w:>7}x{h:<7} {placed:>15} {dpi:>9}  "
              f"{info['colorspace']} .{info['ext']}")

doc.close()
