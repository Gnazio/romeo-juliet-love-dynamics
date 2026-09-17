"""Heart-shaped matplotlib markers.

The outline is the usual cardioid-ish parametric curve

    x = 16 sin^3 t
    y = 13 cos t - 5 cos 2t - 2 cos 3t - cos 4t

re-centred on its own bounding box and scaled so the marker behaves like any
other matplotlib marker: area ~ ``markersize**2``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

__all__ = ["heart_outline", "heart_path", "broken_heart_path"]

_POINTS = 200


def heart_outline(points: int = _POINTS) -> np.ndarray:
    """``(points, 2)`` array tracing a unit heart centred on the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, points)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    xy = np.column_stack([x, y])
    xy -= (xy.max(axis=0) + xy.min(axis=0)) / 2.0  # centre the bounding box
    xy /= np.abs(xy).max()  # fit inside the unit square
    return xy


def heart_path(points: int = _POINTS) -> Path:
    """A closed :class:`~matplotlib.path.Path` usable as ``marker=``."""
    return Path(heart_outline(points), closed=True)


def broken_heart_path(points: int = _POINTS) -> Path:
    """The same heart with a jagged crack down the middle.

    Used to mark feelings that have gone negative -- loathing rather than love.
    """
    outline = heart_outline(points)
    left = outline[outline[:, 0] <= 0]
    right = outline[outline[:, 0] > 0]
    crack_y = np.linspace(1.0, -1.0, 7)
    crack_x = np.array([0.0, 0.16, -0.12, 0.14, -0.10, 0.12, 0.0])
    crack = np.column_stack([crack_x, crack_y])

    vertices, codes = [], []
    for piece, offset in ((left, -0.06), (right, 0.06)):
        if len(piece) < 3:
            continue
        half = np.vstack([piece + (offset, 0.0), crack + (offset, 0.0)])
        vertices.append(half)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(half) - 2) + [Path.CLOSEPOLY])
    if not vertices:  # pragma: no cover - degenerate point counts only
        return heart_path(points)
    return Path(np.vstack(vertices), codes)
