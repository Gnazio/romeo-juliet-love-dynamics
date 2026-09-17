"""Ready-made relationships.

The taxonomy of love affairs as linear systems is Steven Strogatz's; the
nicknames here are our own. See the README for the references.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import LoveAffair

__all__ = ["Preset", "PRESETS", "by_name", "DEFAULT"]


@dataclass(frozen=True)
class Preset:
    name: str
    affair: LoveAffair
    start: tuple[float, float]
    story: str


PRESETS: tuple[Preset, ...] = (
    Preset(
        name="Fickle Romeo & eager Juliet",
        affair=LoveAffair(a=0.0, b=-1.0, c=1.0, d=0.0),
        start=(1.0, 0.0),
        story=(
            "The textbook case. Romeo cools the moment Juliet warms; Juliet warms "
            "the moment Romeo does. M is a 90-degree rotation, so x = cos t and "
            "y = sin t: a perfect circle they never get off."
        ),
    ),
    Preset(
        name="Two cautious lovers",
        affair=LoveAffair(a=-1.0, b=2.0, c=2.0, d=-1.0),
        start=(1.0, -0.4),
        story=(
            "Both pull back from their own feelings but respond strongly to each "
            "other. A saddle: start in the right mood and they escalate into a love "
            "fest, start in the wrong one and it becomes a war."
        ),
    ),
    Preset(
        name="Fire and ice",
        affair=LoveAffair(a=1.0, b=-1.0, c=1.0, d=-1.0),
        start=(0.2, 1.0),
        story=(
            "He runs hot, she runs cold, and the cross-terms cancel the trace "
            "exactly. A degenerate case: the feelings drift along a single line "
            "instead of settling at zero."
        ),
    ),
    Preset(
        name="Peas in a pod",
        affair=LoveAffair(a=1.0, b=1.0, c=1.0, d=1.0),
        start=(0.3, -0.1),
        story=(
            "Identical, mutually reinforcing, no restraint anywhere. Whatever they "
            "start with grows without bound: mutual adoration or mutual loathing, "
            "decided by the first sum x0 + y0."
        ),
    ),
    Preset(
        name="Doomed from the start",
        affair=LoveAffair(a=-0.4, b=-1.5, c=1.5, d=-0.4),
        start=(1.0, 0.0),
        story=(
            "The same carousel as the classic model, but with a little damping on "
            "each of them. A stable spiral: the swings get smaller every lap until "
            "they end in polite indifference."
        ),
    ),
    Preset(
        name="Escalating rollercoaster",
        affair=LoveAffair(a=0.25, b=-1.5, c=1.5, d=0.25),
        start=(0.3, 0.0),
        story=(
            "The carousel again, now with each of them feeding their own feelings. "
            "An unstable spiral: the same cycle of adoration and loathing, louder "
            "every time round."
        ),
    ),
    Preset(
        name="Romeo the robot",
        affair=LoveAffair(a=0.0, b=0.0, c=1.0, d=0.0),
        start=(0.6, -1.0),
        story=(
            "Romeo never changes his mind about anything. Juliet's feelings then "
            "just accumulate his constant regard, linearly, forever."
        ),
    ),
    Preset(
        name="Out of touch with their own feelings",
        affair=LoveAffair(a=0.0, b=1.0, c=1.0, d=0.0),
        start=(1.0, -0.8),
        story=(
            "Neither reacts to their own state, only to the other's. A saddle whose "
            "two eigendirections are the diagonals: on x = y they run away together, "
            "on x = -y they run away from each other."
        ),
    ),
)

DEFAULT = PRESETS[0]


def by_name(name: str) -> Preset:
    """Look up a preset by its display name (case-insensitive)."""
    wanted = name.strip().lower()
    for preset in PRESETS:
        if preset.name.lower() == wanted:
            return preset
    known = ", ".join(repr(p.name) for p in PRESETS)
    raise KeyError(f"unknown preset {name!r}; try one of: {known}")
