"""Extract embedded images at native resolution."""
import fitz

PDF = "/mnt/d/Books/Thesis/My thesis/j.enggeo.2014.08.005.pdf"
PAGES = range(2, 10)

doc = fitz.open(PDF)
for i in PAGES:
    for img in doc[i].get_images(full=True):
        xref = img[0]
        info = doc.extract_image(xref)
        out = f"figures/native_p{i}_x{xref}.{info['ext']}"
        with open(out, "wb") as f:
            f.write(info["image"])
        print(f"{out}   {info['width']}x{info['height']}")
doc.close()
