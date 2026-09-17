"""Romeo & Juliet: love as a 2x2 linear system of differential equations.

    >>> from romeo_juliet import LoveAffair
    >>> affair = LoveAffair(a=0, b=-1, c=1, d=0)   # he is fickle, she is eager
    >>> affair.classify().name
    'Centre'
    >>> affair.solve([1.0, 0.0], [0.0, 3.14159265358979]).round(3).tolist()
    [[1.0, 0.0], [-1.0, 0.0]]
"""

from .model import Classification, LoveAffair, rk4_trajectory
from .plotting import LoveFigure, Simulation
from .presets import DEFAULT, PRESETS, Preset, by_name

__version__ = "1.0.0"
__all__ = [
    "Classification",
    "LoveAffair",
    "LoveFigure",
    "Preset",
    "PRESETS",
    "DEFAULT",
    "Simulation",
    "by_name",
    "rk4_trajectory",
    "__version__",
]
