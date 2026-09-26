r"""Fig. 2 -- PINN architecture and training schematic, with every number read
from a checkpoint's cfg and the loss-term list read from `src.weighting.TERMS`,
so the schematic cannot drift from the network that produced the results.

    PYTHONPATH=. python scripts/make_fig2.py runs/prod_baseline_v2/ckpt_final.pt --out docs/figs/fig2.png

Draw it from the checkpoint the results come from (D-5.6: the production
baseline, not baseline-v1).

Style: `src/plot_style.py`. Drawn at the full 6.25 in text width with 12 pt
text; `draw` refuses to write a figure in which any label spills out of its
box. Writes PNG (300 dpi) and PDF.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

from src import plot_style as ps
from src.weighting import TERMS


#: Box colour families: data in/out, network, physics, optimisation.
FAMILY = {"inputs": ps.COBALT, "outputs": ps.COBALT,
          "net": ps.PURPLE, "ansatz": ps.PURPLE,
          "autograd": ps.VERMILION, "losses": ps.VERMILION,
          "balancer": ps.EMERALD, "optim": ps.EMERALD}
TINT = {ps.COBALT: "#E8EEFD", ps.PURPLE: "#F1EAFE", ps.VERMILION: "#FDECE7",
        ps.EMERALD: "#E3F6EF"}


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def describe(cfg: dict) -> dict:
    """Text for each box, 'Heading\nbody'. Missing keys raise: a schematic
    with guessed numbers is worse than none."""
    need = ("layers", "width", "ansatz", "eps_psi", "n_pde", "n_bc", "epochs")
    miss = [k for k in need if k not in cfg]
    if miss:
        raise KeyError(f"checkpoint cfg lacks {miss}")
    lb = cfg.get("lbfgs_epochs") or 0
    terms = list(TERMS)
    pairs = [", ".join(terms[i:i + 2]) for i in range(0, len(terms), 2)]
    ans = [f"mode = {cfg['ansatz']}",
           rf"$\varepsilon_\psi$ = {_fmt(cfg['eps_psi'])}"]
    if cfg.get("eps_uv") is not None:
        ans.append(rf"$\varepsilon_{{uv}}$ = {_fmt(cfg['eps_uv'])}")
    boxes = {
        "inputs": ("Inputs", "$x^*,\\ z^*,\\ t^*$\n(non-dimensional)"),
        "net": ("Network", f"fully connected\n{cfg['layers']} x {cfg['width']}, tanh"),
        "ansatz": ("Ansatz", "\n".join(ans)),
        "outputs": ("Outputs", "$\\psi$ (head)\n$u,\\ v$ (displ.)"),
        "autograd": ("Autograd", "1st and 2nd\nderivatives"),
        "losses": ("Loss terms",
                   "\n".join(pairs)
                   + "\n+ $w_{\\mathrm{yield}}\\,\\mathcal{L}_{\\mathrm{yield}}$ (SSR)"),
        "balancer": ("Loss balancer",
                     f"weights $w_i$, update every {cfg.get('every', '?')}, "
                     f"$\\lambda$ = {_fmt(cfg.get('lam', '?'))}"),
        "optim": ("Optimiser",
                  f"Adam {cfg['epochs']:,} epochs"
                  + (f"\n→ L-BFGS {lb}" if lb else "")
                  + f"\n$N_{{\\mathrm{{PDE}}}}$ = {cfg['n_pde']:,}"
                  + f"\n$N_{{\\mathrm{{BC}}}}$ = {cfg['n_bc']:,}"),
    }
    return {k: f"{h}\n{b}" for k, (h, b) in boxes.items()}


#: Canvas width: the text block less the 2 x 0.05 in the tight crop adds.
CANVAS_W = ps.WIDTH - 2 * ps.PAD
#: (x, y, w, h) in inches on the CANVAS_W x CANVAS_H canvas. Row 1 is the forward
#: pass, row 2 the physics and the optimiser, row 3 the balancer.
W_BOX = (CANVAS_W - 3 * 0.24) / 4
BOXES = {"inputs": (0.00, 3.08, W_BOX, 1.04),
         "net": (W_BOX + 0.24, 3.08, W_BOX, 1.04),
         "ansatz": (2 * (W_BOX + 0.24), 3.08, W_BOX, 1.04),
         "outputs": (3 * (W_BOX + 0.24), 3.08, W_BOX, 1.04),
         "optim": (0.00, 1.40, 2.02, 1.18),
         "losses": (2.34, 1.08, 2.30, 1.52),
         "autograd": (3 * (W_BOX + 0.24), 1.51, W_BOX, 0.96),
         "balancer": (0.90, 0.02, 4.45, 0.62)}
CANVAS_H = 4.14
INSET = 0.04        # the drawn patch sits this far inside its (x, y, w, h)


def draw(text: dict, out: str):
    ps.apply()
    fig = plt.figure(figsize=(CANVAS_W, CANVAS_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.axis("off")
    texts = {}
    for k, (x, y, w, h) in BOXES.items():
        col = FAMILY[k]
        ax.add_patch(FancyBboxPatch((x + INSET, y + INSET), w - 2 * INSET,
                                    h - 2 * INSET,
                                    boxstyle="round,pad=0.04,rounding_size=0.10",
                                    fc=TINT[col], ec=col, lw=1.4, zorder=2))
        head, body = text[k].split("\n", 1)
        if k == "balancer":          # one-line box: heading and body side by side
            t = ax.text(x + w / 2, y + h / 2, f"{head}: {body}", ha="center",
                        va="center", zorder=3)
            texts[k] = [t]
            continue
        t1 = ax.text(x + w / 2, y + h - 0.12, head, ha="center", va="top",
                     fontweight="bold", color=col, zorder=3)
        t2 = ax.text(x + w / 2, y + h - 0.40, body, ha="center", va="top",
                     linespacing=1.15, zorder=3)
        texts[k] = [t1, t2]

    def arrow(pa, pb, rad=0.0):
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=13,
                                     lw=1.3, color=ps.INK, zorder=4,
                                     shrinkA=0, shrinkB=0,
                                     connectionstyle=f"arc3,rad={rad}"))

    def mid_y(k):
        x, y, w, h = BOXES[k]
        return y + h / 2

    r1 = mid_y("inputs")
    for a_, b_ in (("inputs", "net"), ("net", "ansatz"), ("ansatz", "outputs")):
        xa, _, wa, _ = BOXES[a_]
        xb = BOXES[b_][0]
        arrow((xa + wa - 0.02, r1), (xb + 0.02, r1))
    xo, yo, wo, ho = BOXES["outputs"]
    xg, yg, wg, hg = BOXES["autograd"]
    arrow((xo + wo / 2, yo + 0.02), (xg + wg / 2, yg + hg - 0.02))
    xl, yl, wl, hl = BOXES["losses"]
    arrow((xg + 0.02, mid_y("autograd")), (xl + wl - 0.02, mid_y("autograd")))
    xp, yp, wp, hp = BOXES["optim"]
    arrow((xl + 0.02, yp + hp / 2), (xp + wp - 0.02, yp + hp / 2))
    xb, yb, wb, hb = BOXES["balancer"]
    arrow((xl + wl / 2, yb + hb - 0.02), (xl + wl / 2, yl + 0.02))
    xn, yn, wn, hn = BOXES["net"]
    xa_ = xn + 0.35
    arrow((xa_, yp + hp - 0.02), (xa_, yn + 0.02))
    ax.text(xa_ - 0.08, (yp + hp + yn) / 2, r"update $\theta$", ha="right",
            va="center", zorder=3)
    ax.text(xl - 0.16, yp + hp / 2 + 0.10, r"$\mathcal{L}$", ha="center",
            va="bottom", zorder=3)

    # Every text must sit inside its box: at 12 pt there is no slack, and a
    # label spilling over a border is the defect this check exists for.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for k, ts in texts.items():
        x, y, w, h = BOXES[k]
        box = ax.transData.transform([(x + INSET, y + INSET),
                                      (x + w - INSET, y + h - INSET)])
        for t in ts:
            e = t.get_window_extent(r)
            if (e.x0 < box[0][0] or e.x1 > box[1][0]
                    or e.y0 < box[0][1] or e.y1 > box[1][1]):
                bad.append(k)
    if bad:
        raise RuntimeError(f"fig2: text overflows box(es) {sorted(set(bad))}")
    for p in ps.save(fig, out, fit=False):
        print(f"wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="docs/figs/fig2.png",
                    help="PNG path; a PDF is written next to it")
    a = ap.parse_args(argv)
    cfg = torch.load(a.ckpt, map_location="cpu", weights_only=False)["cfg"]
    draw(describe(cfg), a.out)
    print(f"wrote {a.out}  (from {a.ckpt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
