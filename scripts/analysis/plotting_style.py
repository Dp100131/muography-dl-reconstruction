"""Homogeneous publication-quality figure style for all thesis analysis.

Figure text (labels, titles) is Spanish per the project language contract;
code identifiers stay English.
"""

import matplotlib

matplotlib.use("Agg")  # headless — figures are written to disk only

import matplotlib.pyplot as plt  # noqa: E402


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "image.cmap": "viridis",
    })


def watermark_surrogate(fig) -> None:
    """Every surrogate-engine figure carries the provenance mark (PA-2)."""
    fig.text(0.995, 0.005, "motor surrogate v0.9 (no Geant4)",
             ha="right", va="bottom", fontsize=7, color="0.45")
