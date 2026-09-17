r"""Fig. 2 -- PINN architecture and training schematic, with every number read
from a checkpoint's cfg and the loss-term list read from `src.weighting.TERMS`,
so the schematic cannot drift from the network that produced the results.

    PYTHONPATH=. python scripts/make_fig2.py runs/prod_baseline_v2/ckpt_final.pt --out docs/figs/fig2.png

Draw it from the checkpoint the results come from (D-5.6: the production
baseline, not baseline-v1).
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

from src.weighting import TERMS


def describe(cfg: dict) -> dict:
    """Text for each box. Missing keys raise: a schematic with guessed
    numbers is worse than none."""
    need = ("layers", "width", "ansatz", "eps_psi", "n_pde", "n_bc", "epochs")
    miss = [k for k in need if k not in cfg]
    if miss:
        raise KeyError(f"checkpoint cfg lacks {miss}")
    lb = cfg.get("lbfgs_epochs") or 0
    return {
        "inputs": "inputs\n(x*, z*, t*)\nnon-dimensional",
        "net": f"fully connected\n{cfg['layers']} x {cfg['width']}, tanh",
        "ansatz": f"near-physical ansatz\nmode = {cfg['ansatz']}\n"
                  f"eps_psi = {cfg['eps_psi']}",
        "outputs": "outputs\npsi (head), u, v",
        "autograd": "autograd\n1st and 2nd derivatives",
        "losses": "loss terms\n" + "\n".join(TERMS) + "\n+ w_yield * L_yield (SSR)",
        "balancer": f"loss balancer\nupdate every {cfg.get('every', '?')}, "
                    f"lambda {cfg.get('lam', '?')}",
        "optim": f"Adam {cfg['epochs']:,} epochs"
                 + (f"\n-> L-BFGS {lb}" if lb else "")
                 + f"\nN_PDE {cfg['n_pde']:,}, N_BC {cfg['n_bc']:,}",
    }


def draw(text: dict, out: str, dpi=200):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = {"inputs": (0.2, 2.6, 1.6, 1.3), "net": (2.3, 2.6, 1.9, 1.3),
             "ansatz": (4.7, 2.6, 2.0, 1.3), "outputs": (7.2, 2.6, 1.7, 1.3),
             "autograd": (7.2, 0.4, 1.7, 1.3), "losses": (4.4, 0.1, 2.5, 2.1),
             "balancer": (2.1, 0.4, 2.0, 1.3), "optim": (9.2, 1.4, 1.7, 1.7)}
    ax.set_ylim(0, 5.2)
    for k, (x, y, w, h) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc="#eef3f8", ec="#34506b", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text[k], ha="center", va="center",
                fontsize=7.5)

    def arrow(pa, pb, label=None, rad=0.0):
        ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="-|>", mutation_scale=12,
                                     lw=1.1, color="#34506b",
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2 + 0.12, label,
                    ha="center", fontsize=6.5, color="#34506b")
    arrow((1.85, 3.25), (2.25, 3.25))                        # inputs -> net
    arrow((4.25, 3.25), (4.65, 3.25))                        # net -> ansatz
    arrow((6.75, 3.25), (7.15, 3.25))                        # ansatz -> outputs
    arrow((8.05, 2.55), (8.05, 1.75))                        # outputs -> autograd
    arrow((7.15, 1.05), (6.95, 1.05))                        # autograd -> losses
    arrow((4.15, 1.05), (4.35, 1.05), "weights")             # balancer -> losses
    arrow((6.95, 1.9), (9.15, 2.0), "weighted total")        # losses -> optimiser
    arrow((10.05, 3.15), (3.25, 3.95), "update parameters", rad=0.25)  # optim -> net
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="docs/figs/fig2.png")
    a = ap.parse_args(argv)
    cfg = torch.load(a.ckpt, map_location="cpu", weights_only=False)["cfg"]
    draw(describe(cfg), a.out)
    print(f"wrote {a.out}  (from {a.ckpt})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
