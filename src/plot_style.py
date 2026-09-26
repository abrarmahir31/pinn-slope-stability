r"""One house style for every thesis figure.

    from src import plot_style as ps
    ps.apply()                                   # once, before any figure
    fig, ax = plt.subplots(figsize=(ps.WIDTH, 3.4))
    ...
    ps.save(fig, "docs/figs/fig4.png")           # writes fig4.png AND fig4.pdf

THE RULES (thesis formatting guideline, A4 portrait, 1-inch margins)
  * Times New Roman, 12 pt, for EVERY text element; mathtext set to STIX so
    symbols match the Times body text.
  * Width 6.25 in (the ~158-160 mm text block). Side-by-side panels share that
    budget; nothing is ever drawn wider and shrunk in Word, which is what
    keeps 12 pt in the figure equal to 12 pt in the body.
  * Vector PDF + 300 dpi PNG, `bbox_inches="tight"`, `pad_inches=0.05`.
    `save` measures the written PNG and warns if the trimmed figure came out
    wider than the text block.
  * One colour rule across the thesis:
        criterion  -> colour   (GHB cobalt, MC vermilion)      CRITERION
        stratum    -> colour   (Mk amber, Mk_d purple, Tm teal) STRATA
        w_yield    -> marker / line style                       W_MARKER
    Loss terms, boundary segments and sensitivity levels have their own fixed
    maps below; a figure never picks colours ad hoc.

Times New Roman is not installed on every Linux box. The fallback chain goes
to metric-compatible Times clones (TeX Gyre Termes, Liberation Serif, STIX),
so line breaks and layout are the same on the lab PC and here.
"""
from __future__ import annotations

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- page
MM = 1.0 / 25.4
WIDTH = 6.25            # in, full text-block width (A4, 1-in margins)
WIDTH_MAX = 6.30        # in, what `save` tolerates after the tight crop
HALF = 3.05             # in, one of two side-by-side figures
FONT_PT = 12
DPI = 300
PAD = 0.05

SERIF = ["Times New Roman", "Times", "TeX Gyre Termes", "Liberation Serif",
         "Nimbus Roman", "STIXGeneral", "DejaVu Serif"]

# --------------------------------------------------------------------------- colour
COBALT = "#1D4ED8"
VERMILION = "#E4572E"
AMBER = "#F59E0B"
EMERALD = "#059669"
PURPLE = "#7C3AED"
CRIMSON = "#C81D25"
TEAL = "#0E9AA7"
INK = "#111827"
GREY = "#6B7280"
LIGHT = "#E5E7EB"

CATEGORICAL = [COBALT, VERMILION, AMBER, EMERALD, PURPLE, INK, TEAL, CRIMSON]

#: Failure criterion. The same two colours in Figs 8, 10, S1, S2 and 9.
CRITERION = {"GHB": COBALT, "MC": VERMILION}

#: Strata: vivid for lines/points, light tint for area fills.
STRATA = {"Mk": AMBER, "Mk_d": PURPLE, "Tm": TEAL}
STRATA_FILL = {"Mk": "#FCE7B2", "Mk_d": "#DCCCFB", "Tm": "#BFE8EC"}
STRATA_LABEL = {"Mk": "Mk", "Mk_d": r"Mk$_\mathrm{d}$", "Tm": "Tm"}

#: w_yield -> marker (colour stays the criterion's).
W_MARKER = {0.3: "v", 1.0: "o", 3.0: "^"}

#: Unweighted loss terms (Fig 4).
LOSS = {"bc_mech": INK, "pde_mech": CRIMSON, "pde_richards": COBALT,
        "bc": EMERALD, "ic_head": AMBER, "ic_disp": PURPLE, "interface": GREY}
LOSS_STYLE = {"bc_mech": "-", "pde_mech": "-", "pde_richards": "-",
              "bc": "--", "ic_head": "--", "ic_disp": "-.", "interface": ":"}

#: Boundary segments (Fig 3). Chosen to stand off the strata tints.
SEGMENT = {"natural_ground": EMERALD, "bench": "#DB2777", "cut_face": CRIMSON,
           "pit_floor": "#92400E", "base": INK, "far_field_f1": COBALT}

#: Sensitivity levels (Fig 12): low / high end of each factor's range.
LEVEL = {"low": PURPLE, "high": AMBER}

#: Continuous fields.
CMAP_FIELD = "viridis"        # psi, sequential
CMAP_DIVERGING = "RdBu_r"     # departures, symmetric about zero
CMAP_MAGNITUDE = "turbo"      # |d|, gamma_max
CMAP_FS = "RdYlBu"            # FS: red < 1 < blue, centred on 1


def apply() -> None:
    """Set the house rcParams. Idempotent."""
    matplotlib.use("Agg", force=False)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "mathtext.default": "it",
        "font.size": FONT_PT,
        "axes.labelsize": FONT_PT,
        "axes.titlesize": FONT_PT,
        "axes.titleweight": "normal",
        "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT,
        "legend.fontsize": FONT_PT,
        "legend.title_fontsize": FONT_PT,
        "figure.titlesize": FONT_PT,
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": PAD,
        "pdf.fonttype": 42,           # embed TrueType, text stays selectable
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": LIGHT,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "patch.linewidth": 0.9,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": LIGHT,
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
        "image.cmap": CMAP_FIELD,
        "figure.constrained_layout.use": False,
    })


def field_axes(ax) -> None:
    """Axes that carry an image or contour field: no grid over the data."""
    ax.grid(False)
    ax.set_axisbelow(False)


def panel_label(ax, letter: str, inside: bool = True, **kw) -> None:
    """'(a)' in the upper-left corner, 12 pt, on a white chip if inside."""
    if inside:
        ax.text(0.015, 0.975, f"({letter})", transform=ax.transAxes,
                ha="left", va="top", zorder=20,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                          alpha=0.85), **kw)
    else:
        ax.set_title(f"({letter})", loc="left")


def colorbar(fig, mappable, ax, label: str, **kw):
    """Thin colour bar with 12 pt label and inward ticks."""
    kw.setdefault("pad", 0.02)
    kw.setdefault("fraction", 0.05)
    kw.setdefault("extendfrac", 0.035)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.set_label(label)
    cb.outline.set_linewidth(0.8)
    cb.ax.tick_params(direction="in", width=0.8)
    return cb


def fit_width(fig, target: float = WIDTH, iters: int = 6) -> float:
    """Resize `fig` so that its TIGHT-CROPPED width (content + 2 x PAD) is
    `target`. With a layout engine the axes reflow, so labels are never
    clipped and nothing is shrunk; the height is left alone. Returns the
    final cropped width in inches."""
    for _ in range(iters):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        got = bb.width + 2 * PAD
        if abs(got - target) < 0.01:
            return got
        fig.set_figwidth(fig.get_figwidth() + (target - got))
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD


def save(fig, out: str, formats=("png", "pdf"), close: bool = True,
         fit: bool = True) -> list:
    """Write `out` in every format (the extension of `out` is replaced), 300
    dpi, tight, 0.05 in pad. With `fit` (default) the figure is first resized
    so the cropped result is exactly the 6.25 in text width. Returns the
    written paths. Warns if the cropped PNG is wider than the text block."""
    if fit:
        fit_width(fig)
    base, _ = os.path.splitext(out)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    paths = []
    for ext in formats:
        p = f"{base}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=PAD)
        paths.append(p)
    if close:
        plt.close(fig)
    png = f"{base}.png"
    if os.path.exists(png):
        w_in, h_in = png_size_in(png)
        if w_in > WIDTH_MAX + 1e-3:
            warnings.warn(f"{png}: {w_in:.2f} in wide after the tight crop; "
                          f"the text block is {WIDTH:.2f} in", stacklevel=2)
    return paths


def png_size_in(path: str) -> tuple[float, float]:
    """(width, height) in inches of a PNG written at DPI."""
    import struct
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w / DPI, h / DPI


# --------------------------------------------------------------------------- domain
def strata_grid(nx: int = 500, nz: int = 320, x=(None, None), z=(None, None)):
    """(X, Z, inside, code) over the Section 5 domain; `code` is the integer
    index of the stratum in ("Mk", "Mk_d", "Tm"), -1 outside."""
    from src.step21_geometry import geometry as g
    x0 = g.X_MIN if x[0] is None else x[0]
    x1 = g.X_MAX if x[1] is None else x[1]
    xs = np.linspace(x0, x1, nx)
    z0 = g.Z_BASE if z[0] is None else z[0]
    z1 = float(np.max(g.z_ground(xs))) if z[1] is None else z[1]
    zs = np.linspace(z0, z1, nz)
    X, Z = np.meshgrid(xs, zs)
    inside = np.asarray(g.inside_domain(X, Z), bool)
    tag = np.asarray(g.material_tag(X, Z)).astype(str)
    code = np.full(X.shape, -1, int)
    for i, t in enumerate(("Mk", "Mk_d", "Tm")):
        code[inside & (tag == t)] = i
    return X, Z, inside, code


def draw_strata(ax, fill: bool = False, lines: bool = True, lw: float = 0.9,
                color: str = INK, nx: int = 500, nz: int = 320, x=(None, None),
                z=(None, None), label_contacts: bool = False):
    """Overlay stratum contacts (and optionally light fills) on `ax`. Contacts
    are the level sets of the live `geometry.material_tag`, so they cannot
    drift from the model."""
    from matplotlib.colors import ListedColormap
    X, Z, inside, code = strata_grid(nx, nz, x, z)
    if fill:
        cm = ListedColormap([STRATA_FILL[t] for t in ("Mk", "Mk_d", "Tm")])
        ax.pcolormesh(X, Z, np.ma.masked_less(code, 0), cmap=cm, vmin=-0.5,
                      vmax=2.5, shading="auto", zorder=0, rasterized=True)
    if lines:
        c = np.where(inside, code, np.nan)
        ax.contour(X, Z, c, levels=[0.5, 1.5], colors=color, linewidths=lw,
                   linestyles="-", zorder=5)
    return X, Z, inside, code


def ground_line(ax, lw: float = 1.1, color: str = INK, **kw):
    from src.step21_geometry import geometry as g
    xs = np.linspace(g.X_MIN, g.X_MAX, 800)
    ax.plot(xs, g.z_ground(xs), "-", color=color, lw=lw, zorder=6, **kw)


def domain_outline(ax, lw: float = 0.9, color: str = INK, zorder: int = 6):
    """Closed outline of the model domain: ground surface, F1 far-field
    fault, base and the x = 0 pit-floor edge -- all from the live geometry."""
    from src.step21_geometry import geometry as g
    xs = np.linspace(g.X_MIN, g.X_MAX, 1200)
    zt = g.z_ground(xs)
    keep = xs <= g.x_f1(zt)
    xs, zt = xs[keep], zt[keep]
    zf = np.linspace(zt[-1], g.Z_BASE, 50)
    ox = np.concatenate([[g.X_MIN], xs, g.x_f1(zf), [g.X_MIN, g.X_MIN]])
    oz = np.concatenate([[zt[0]], zt, zf, [g.Z_BASE, zt[0]]])
    ax.plot(ox, oz, "-", color=color, lw=lw, zorder=zorder)


def domain_extent(pad: float = 2.0):
    """(xmin, xmax, zmin, zmax) that frames the whole domain."""
    from src.step21_geometry import geometry as g
    xs = np.linspace(g.X_MIN, g.X_MAX, 800)
    return (g.X_MIN, g.X_MAX, g.Z_BASE - pad,
            float(np.max(g.z_ground(xs))) + pad)


def masked_triangulation(x, z, max_edge: float | None = None):
    """Delaunay triangulation of scattered domain points with every triangle
    whose centroid lies outside the (concave) domain removed, so a filled
    contour never bridges the excavation."""
    import matplotlib.tri as mtri
    from src.step21_geometry import geometry as g
    tri = mtri.Triangulation(x, z)
    t = tri.triangles
    cx, cz = x[t].mean(1), z[t].mean(1)
    mask = ~np.asarray(g.inside_domain(cx, cz), bool)
    if max_edge is not None:
        e = np.max(np.stack([np.hypot(x[t[:, i]] - x[t[:, (i + 1) % 3]],
                                      z[t[:, i]] - z[t[:, (i + 1) % 3]])
                             for i in range(3)]), axis=0)
        mask |= e > max_edge
    tri.set_mask(mask)
    return tri
