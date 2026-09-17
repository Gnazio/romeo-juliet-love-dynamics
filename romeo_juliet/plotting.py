"""The three panels, and the machinery that keeps them in sync.

``LoveFigure`` owns a matplotlib figure with

* a **phase portrait** -- the vector field of ``v' = M v`` plus the orbit the
  couple actually travels, with a heart riding the leading edge;
* a **time series** -- ``x(t)`` and ``y(t)``, each with its own heart marker
  sliding along the curve;
* a **heartbeat panel** -- Romeo's heart and Juliet's heart, drawn large, each
  one sized by how strongly that person feels and cracked open when the feeling
  has turned to loathing.

Both the Tk application and the headless renderer drive the same object: build
it once, then call :meth:`LoveFigure.set_frame` for each step in time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hearts import broken_heart_path, heart_path
from .model import LoveAffair
from .theme import FIELD_CMAP, PALETTE

__all__ = ["Simulation", "LoveFigure"]

_HEART = heart_path()
_BROKEN = broken_heart_path()
_BLOWUP = 8.0  # stop following a trajectory once it is this far from the origin


@dataclass(frozen=True)
class Simulation:
    """A solved affair: the times, the states, and a sensible viewport."""

    affair: LoveAffair
    start: np.ndarray
    times: np.ndarray
    states: np.ndarray
    truncated: bool = False

    @classmethod
    def run(
        cls,
        affair: LoveAffair,
        start,
        duration: float = 24.0,
        samples: int = 1400,
    ) -> "Simulation":
        start = np.asarray(start, dtype=float)
        times = np.linspace(0.0, duration, samples)
        states = affair.solve(start, times)

        # Runaway solutions overflow the plot (and eventually float64); follow
        # them only as far as they stay interesting to look at.
        radius = np.linalg.norm(states, axis=1)
        cap = _BLOWUP * max(np.linalg.norm(start), 1.0)
        beyond = np.flatnonzero(~np.isfinite(radius) | (radius > cap))
        if beyond.size:
            keep = max(int(beyond[0]) + 1, 10)
            times, states = times[:keep], states[:keep]
        return cls(affair=affair, start=start, times=times, states=states,
                   truncated=bool(beyond.size))

    @property
    def size(self) -> int:
        return len(self.times)

    def limits(self) -> float:
        """Half-width of a square viewport that frames the orbit."""
        extent = float(np.abs(self.states).max(initial=0.0))
        extent = max(extent, float(np.abs(self.start).max()), 0.5)
        return extent * 1.18


class LoveFigure:
    """Draws one :class:`Simulation` and animates a cursor through it."""

    def __init__(self, figure, *, show_field: bool = True, show_eigen: bool = True):
        self.figure = figure
        self.show_field = show_field
        self.show_eigen = show_eigen
        self.simulation: Simulation | None = None

        grid = figure.add_gridspec(
            2, 2, width_ratios=[1.06, 1.0], height_ratios=[1.25, 1.0],
            left=0.07, right=0.975, top=0.88, bottom=0.09, wspace=0.24, hspace=0.42,
        )
        self.ax_phase = figure.add_subplot(grid[:, 0])
        self.ax_time = figure.add_subplot(grid[0, 1])
        self.ax_hearts = figure.add_subplot(grid[1, 1])

        self._build_phase()
        self._build_time()
        self._build_hearts()

    # ------------------------------------------------------------ scaffolding
    def _build_phase(self) -> None:
        ax = self.ax_phase
        ax.set_title("Phase portrait")
        ax.set_xlabel("x  —  Romeo's love for Juliet")
        ax.set_ylabel("y  —  Juliet's love for Romeo")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.5)
        ax.axhline(0, color=PALETTE["grid"], lw=1.0, zorder=1)
        ax.axvline(0, color=PALETTE["grid"], lw=1.0, zorder=1)

        self._field = None
        self._nullclines = [
            ax.plot([], [], ls=(0, (6, 4)), lw=1.1, color=PALETTE["romeo"], alpha=0.45,
                    zorder=3, label="dx/dt = 0")[0],
            ax.plot([], [], ls=(0, (6, 4)), lw=1.1, color=PALETTE["juliet"], alpha=0.45,
                    zorder=3, label="dy/dt = 0")[0],
        ]
        self._eigenlines = [
            ax.plot([], [], lw=1.4, color=PALETTE["ink"], alpha=0.35, zorder=3)[0]
            for _ in range(2)
        ]
        (self._future,) = ax.plot([], [], lw=1.4, color=PALETTE["orbit"], alpha=0.55,
                                  zorder=4)
        (self._past,) = ax.plot([], [], lw=2.6, color=PALETTE["accent"], zorder=5)
        (self._origin,) = ax.plot([0], [0], marker="o", ms=5, color=PALETTE["ink"],
                                  alpha=0.55, zorder=6)
        (self._start,) = ax.plot([], [], marker="o", ms=6, mfc="none", mew=1.6,
                                 color=PALETTE["ink"], alpha=0.6, zorder=6)
        (self._cursor,) = ax.plot([], [], marker=_HEART, ms=17, color=PALETTE["romeo"],
                                  mec=PALETTE["ink"], mew=0.6, ls="none", zorder=7)
        self._note = ax.text(0.015, 0.015, "", transform=ax.transAxes, fontsize=8,
                             style="italic", color=PALETTE["muted"], zorder=8)

    def _build_time(self) -> None:
        ax = self.ax_time
        ax.set_xlabel("time")
        ax.grid(True, alpha=0.5)
        ax.axhline(0, color=PALETTE["grid"], lw=1.0)

        (self._romeo_line,) = ax.plot([], [], lw=2.0, color=PALETTE["romeo"],
                                      label="Romeo  x(t)")
        (self._juliet_line,) = ax.plot([], [], lw=2.0, color=PALETTE["juliet"],
                                       label="Juliet  y(t)")
        (self._romeo_ghost,) = ax.plot([], [], lw=1.0, color=PALETTE["romeo"], alpha=0.2)
        (self._juliet_ghost,) = ax.plot([], [], lw=1.0, color=PALETTE["juliet"], alpha=0.2)
        (self._romeo_dot,) = ax.plot([], [], marker=_HEART, ms=13, ls="none",
                                     color=PALETTE["romeo"], mec="white", mew=0.7, zorder=6)
        (self._juliet_dot,) = ax.plot([], [], marker=_HEART, ms=13, ls="none",
                                      color=PALETTE["juliet"], mec="white", mew=0.7, zorder=6)
        # Above the axes, so a tall curve never has to share space with it.
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncols=2,
                  handlelength=1.6, columnspacing=1.6)
        ax.set_title("Feelings over time", pad=22)

    def _build_hearts(self) -> None:
        ax = self.ax_hearts
        ax.set_title("Two hearts")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        (self._romeo_heart,) = ax.plot([-0.45], [0.08], marker=_HEART, ms=40, ls="none",
                                       color=PALETTE["romeo"], mec=PALETTE["ink"], mew=0.8)
        (self._juliet_heart,) = ax.plot([0.45], [0.08], marker=_HEART, ms=40, ls="none",
                                        color=PALETTE["juliet"], mec=PALETTE["ink"], mew=0.8)
        self._romeo_text = ax.text(-0.45, -0.78, "", ha="center", va="center", fontsize=8.5,
                                   color=PALETTE["ink"])
        self._juliet_text = ax.text(0.45, -0.78, "", ha="center", va="center", fontsize=8.5,
                                    color=PALETTE["ink"])
        self._verdict = ax.text(0.0, 0.92, "", ha="center", va="top", fontsize=9,
                                style="italic", color=PALETTE["muted"])

    # ---------------------------------------------------------------- content
    def set_simulation(self, simulation: Simulation) -> None:
        """Swap in a new affair and redraw everything that does not move."""
        self.simulation = simulation
        affair = simulation.affair
        span = simulation.limits()

        self.ax_phase.set_xlim(-span, span)
        self.ax_phase.set_ylim(-span, span)
        self._draw_field(affair, span)
        self._draw_nullclines(affair, span)
        self._draw_eigenlines(affair, span)

        self._future.set_data(simulation.states[:, 0], simulation.states[:, 1])
        self._start.set_data([simulation.start[0]], [simulation.start[1]])
        self._romeo_ghost.set_data(simulation.times, simulation.states[:, 0])
        self._juliet_ghost.set_data(simulation.times, simulation.states[:, 1])

        self.ax_time.set_xlim(simulation.times[0], max(simulation.times[-1], 1e-6))
        reach = max(float(np.abs(simulation.states).max(initial=0.0)) * 1.08, 0.5)
        self.ax_time.set_ylim(-reach, reach)

        self._note.set_text(
            f"feelings run off the chart at t ≈ {simulation.times[-1]:.1f}"
            if simulation.truncated else ""
        )

        eq_x, eq_y = affair.equations()
        self.ax_phase.set_title(f"Phase portrait      {eq_x},   {eq_y}")
        self.set_frame(0)

    def _draw_field(self, affair: LoveAffair, span: float) -> None:
        if self._field is not None:
            self._field.remove()
            self._field = None
        if not self.show_field:
            return
        axis = np.linspace(-span, span, 17)
        gx, gy = np.meshgrid(axis, axis)
        u = affair.a * gx + affair.b * gy
        v = affair.c * gx + affair.d * gy
        speed = np.hypot(u, v)
        safe = np.where(speed > 1e-12, speed, 1.0)
        self._field = self.ax_phase.quiver(
            gx, gy, u / safe, v / safe, speed,
            cmap=FIELD_CMAP, pivot="mid", scale=26, width=0.0042,
            headwidth=3.4, headlength=4.0, alpha=0.85, zorder=2,
        )

    def _draw_nullclines(self, affair: LoveAffair, span: float) -> None:
        for line, (px, py) in zip(
            self._nullclines, ((affair.a, affair.b), (affair.c, affair.d))
        ):
            line.set_data(*_line_through_origin(px, py, span))

    def _draw_eigenlines(self, affair: LoveAffair, span: float) -> None:
        values, vectors = affair.eigenvectors()
        for index, line in enumerate(self._eigenlines):
            visible = (
                self.show_eigen
                and index < len(values)
                and abs(values[index].imag) < 1e-9
            )
            if not visible:
                line.set_data([], [])
                continue
            vec = vectors[:, index].real
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                line.set_data([], [])
                continue
            vec = vec / norm * span * 1.6
            line.set_data([-vec[0], vec[0]], [-vec[1], vec[1]])

    # -------------------------------------------------------------- animation
    def set_frame(self, index: int) -> None:
        """Move every moving thing to sample ``index`` of the simulation."""
        sim = self.simulation
        if sim is None:
            return
        index = int(np.clip(index, 0, sim.size - 1))
        x, y = sim.states[index]

        self._past.set_data(sim.states[: index + 1, 0], sim.states[: index + 1, 1])
        self._cursor.set_data([x], [y])
        self._cursor.set_markersize(13 + 5 * np.tanh(np.hypot(x, y)))

        self._romeo_line.set_data(sim.times[: index + 1], sim.states[: index + 1, 0])
        self._juliet_line.set_data(sim.times[: index + 1], sim.states[: index + 1, 1])
        self._romeo_dot.set_data([sim.times[index]], [x])
        self._juliet_dot.set_data([sim.times[index]], [y])

        self._beat(self._romeo_heart, self._romeo_text, x, "Romeo", PALETTE["romeo"])
        self._beat(self._juliet_heart, self._juliet_text, y, "Juliet", PALETTE["juliet"])
        self._verdict.set_text(_verdict(x, y))

    @staticmethod
    def _beat(artist, label, value: float, who: str, warm: str) -> None:
        if abs(value) < 5e-3:  # don't print "-0.00" at someone
            value = 0.0
        strength = np.tanh(abs(value))
        artist.set_markersize(18 + 46 * strength)
        artist.set_marker(_HEART if value >= 0 else _BROKEN)
        artist.set_color(warm if value >= 0 else PALETTE["cold"])
        artist.set_alpha(0.32 + 0.68 * strength)
        mood = "loves" if value > 0.05 else "loathes" if value < -0.05 else "is unmoved by"
        other = "Juliet" if who == "Romeo" else "Romeo"
        label.set_text(f"{who} {mood} {other}\n{value:+.2f}")


def _line_through_origin(px: float, py: float, span: float):
    """The set ``px*x + py*y = 0``, clipped to a square of half-width ``span``."""
    reach = span * 1.6
    if abs(px) < 1e-12 and abs(py) < 1e-12:
        return [], []  # every point qualifies; drawing a line would be a lie
    if abs(py) < 1e-12:  # vertical line x = 0
        return [0.0, 0.0], [-reach, reach]
    slope = -px / py
    return [-reach, reach], [-reach * slope, reach * slope]


def _verdict(x: float, y: float) -> str:
    if x > 0.05 and y > 0.05:
        return "requited"
    if x < -0.05 and y < -0.05:
        return "mutual loathing"
    if abs(x) <= 0.05 and abs(y) <= 0.05:
        return "indifference"
    return "unrequited"
