"""Headless rendering: a still image or an animated GIF, no window needed.

    python -m romeo_juliet render --preset "Fickle Romeo & eager Juliet" -o out.png
    python -m romeo_juliet render --params 0 -1 1 0 --gif docs/orbit.gif
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from .model import LoveAffair  # noqa: E402
from .plotting import LoveFigure, Simulation  # noqa: E402
from .presets import Preset  # noqa: E402
from .theme import apply_rcparams  # noqa: E402

__all__ = ["render_still", "render_gif"]


def _figure(preset: Preset, duration: float, samples: int,
            dpi: int = 110) -> tuple[Figure, LoveFigure, Simulation]:
    apply_rcparams(matplotlib.rcParams)
    figure = Figure(figsize=(11.0, 6.4), dpi=dpi)
    view = LoveFigure(figure)
    simulation = Simulation.run(preset.affair, preset.start, duration, samples)
    view.set_simulation(simulation)
    figure.suptitle(
        f"{preset.name}  —  {preset.affair.classify().name}",
        fontsize=13, fontweight="bold", y=0.975,
    )
    return figure, view, simulation


def render_still(
    preset: Preset,
    out: Path,
    *,
    at: float = 1.0,
    duration: float = 24.0,
    samples: int = 1400,
) -> Path:
    """Write a single frame; ``at`` is the fraction of the run to freeze on."""
    figure, view, simulation = _figure(preset, duration, samples)
    view.set_frame(int((simulation.size - 1) * min(max(at, 0.0), 1.0)))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=figure.dpi, facecolor=figure.get_facecolor())
    return out


def render_gif(
    preset: Preset,
    out: Path,
    *,
    frames: int = 120,
    fps: int = 20,
    duration: float = 24.0,
    samples: int = 1400,
    dpi: int = 68,
    colors: int = 64,
) -> Path:
    """Write an animated GIF of the whole affair (needs Pillow).

    The default dpi is deliberately low: a README GIF wants to be a few hundred
    kilobytes, not a few megabytes.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    figure, view, simulation = _figure(preset, duration, samples, dpi=dpi)
    step = max(simulation.size // max(frames, 1), 1)

    animation = FuncAnimation(
        figure,
        lambda i: view.set_frame(min(i * step, simulation.size - 1)),
        frames=frames,
        interval=1000 // fps,
        blit=False,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(out, writer=PillowWriter(fps=fps),
                   savefig_kwargs={"facecolor": figure.get_facecolor()})
    _shrink(out, colors=colors)
    return out


def _shrink(path: Path, *, colors: int = 64) -> None:
    """Re-encode a GIF onto a small shared palette.

    Matplotlib writes full-colour frames; a few dozen colours are plenty for
    flat-shaded plots and cut the file size by an order of magnitude.
    """
    from PIL import Image, ImageSequence

    with Image.open(path) as gif:
        duration = gif.info.get("duration", 50)
        frames = [
            frame.convert("RGB").quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
            for frame in ImageSequence.Iterator(gif)
        ]
    # disposal=1 (leave the previous frame in place) lets Pillow store only the
    # rectangle that actually changed between frames.
    frames[0].save(path, save_all=True, append_images=frames[1:], optimize=True,
                   duration=duration, loop=0, disposal=1)
