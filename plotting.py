"""Shared chart style so every figure in the report looks the same."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# First three slots of a colour-blind-validated categorical palette, always in this order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "text.color": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def points(ax, x, y, color, label=None, size=24, alpha=0.7):
    """Scatter markers with a thin surface-coloured ring so overlapping points stay readable."""
    ax.scatter(x, y, s=size, color=color, alpha=alpha, edgecolors=SURFACE, linewidths=0.8, label=label, zorder=3)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
