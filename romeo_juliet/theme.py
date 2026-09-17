"""Colours and matplotlib defaults shared by the GUI and the renderer."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

__all__ = ["PALETTE", "FIELD_CMAP", "apply_rcparams"]

PALETTE = {
    "bg": "#fdf4f6",
    "panel": "#ffffff",
    "ink": "#3a2c33",
    "muted": "#9a8b93",
    "grid": "#ead9e0",
    "romeo": "#d62839",
    "juliet": "#7b2cbf",
    "orbit": "#c9a6b6",
    "accent": "#ef8fa3",
    "cold": "#5c7a99",
}

# Pale pink to deep plum: used to tint the vector field by flow speed.
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "love_field", ["#f7d9e3", "#e79ab4", "#b8628c", "#6d3a63"]
)


def apply_rcparams(rcparams) -> None:
    """Apply the project look to a matplotlib rcParams mapping."""
    rcparams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 9,
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
        }
    )
