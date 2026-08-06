"""Locate the page containing Fig. 5 (typical cross-sections incl. 5-5')."""
import fitz

PDF = "/mnt/d/Books/Thesis/My thesis/j.enggeo.2014.08.005.pdf"

doc = fitz.open(PDF)
print(f"Total pages: {doc.page_count}\n")
print(f"{'idx':>4} {'printed':>8} {'vectors':>8} {'imgs':>5}  caption hits")
print("-" * 62)

KEYS = ["Fig. 5", "Figure 5", "5-5", "5\u20135", "Typical cross", "cross-section"]

for i, page in enumerate(doc):
    text = page.get_text()
    hits = [k for k in KEYS if k in text]
    n_vec = len(page.get_drawings())
    n_img = len(page.get_images(full=True))
    if hits or n_vec > 200:
        print(f"{i:>4} {i+1:>8} {n_vec:>8} {n_img:>5}  {hits}")

doc.close()
