"""Extract every embedded image with its native size, all pages."""
import fitz
PDF = "/mnt/d/Books/Thesis/My thesis/j.enggeo.2014.08.005.pdf"
doc = fitz.open(PDF)
for i in range(doc.page_count):
    for img in doc[i].get_images(full=True):
        x = img[0]; info = doc.extract_image(x)
        out = f"figures/native_p{i}_x{x}.{info['ext']}"
        with open(out, "wb") as f:
            f.write(info["image"])
        print(f"{out:<40} {info['width']}x{info['height']}")
doc.close()
