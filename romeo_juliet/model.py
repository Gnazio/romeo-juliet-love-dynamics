"""The linear love-affair model.

Two people, two numbers.  ``x(t)`` is Romeo's feeling for Juliet, ``y(t)`` is
Juliet's feeling for Romeo.  Positive means love, negative means loathing, zero
means indifference.  Each person's feelings change at a rate that depends
linearly on both:

.. math::

    \\frac{dx}{dt} = a\\,x + b\\,y \\qquad
    \\frac{dy}{dt} = c\\,x + d\\,y

which is the matrix differential equation

.. math::

    \\frac{d}{dt}\\vec{v}(t) = \\mathbf{M}\\,\\vec{v}(t), \\qquad
    \\mathbf{M} = \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}

The textbook Romeo & Juliet case is ``a=0, b=-1, c=1, d=0``: Romeo cools off
whenever Juliet warms up, Juliet warms up whenever Romeo does.  ``M`` is then a
90-degree rotation matrix, the solution is ``x = cos t``, ``y = sin t``, and the
two of them circle each other forever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LoveAffair", "Classification", "rk4_trajectory"]


@dataclass(frozen=True)
class Classification:
    """What the fixed point at the origin looks like."""

    name: str
    stability: str
    blurb: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.stability})"


@dataclass(frozen=True)
class LoveAffair:
    """A linear love affair ``v' = M v``.

    Parameters
    ----------
    a: how Romeo's love responds to his own love (a>0: he eggs himself on).
    b: how Romeo's love responds to Juliet's (b<0: he is fickle, he flees).
    c: how Juliet's love responds to Romeo's (c>0: she is eager, she follows).
    d: how Juliet's love responds to her own love.
    """

    a: float = 0.0
    b: float = -1.0
    c: float = 1.0
    d: float = 0.0

    # ------------------------------------------------------------------ core
    @property
    def matrix(self) -> np.ndarray:
        """The 2x2 matrix ``M``."""
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=float)

    @property
    def trace(self) -> float:
        return self.a + self.d

    @property
    def determinant(self) -> float:
        return self.a * self.d - self.b * self.c

    @property
    def discriminant(self) -> float:
        """``tr^2 - 4 det``: negative means the couple spirals/rotates."""
        return self.trace**2 - 4.0 * self.determinant

    def derivative(self, state) -> np.ndarray:
        """The velocity ``M v`` at a point (or at a stack of points)."""
        state = np.asarray(state, dtype=float)
        return np.asarray(state @ self.matrix.T if state.ndim > 1 else self.matrix @ state)

    # -------------------------------------------------------------- solution
    def _propagator_coefficients(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(alpha, beta)`` with ``exp(Mt) = alpha*I + beta*(M - mu*I)``.

        Writing ``mu = tr/2`` and ``delta = mu^2 - det``, the Cayley-Hamilton
        theorem collapses the usual infinite series into three cases -- real
        eigenvalues, complex ones, or a repeated one.  No series truncation, no
        integration error: the trajectory is exact to machine precision.
        """
        mu = 0.5 * self.trace
        delta = 0.25 * self.discriminant
        scale = np.exp(mu * t)

        if delta > 1e-12:  # distinct real eigenvalues -> node or saddle
            s = np.sqrt(delta)
            return scale * np.cosh(s * t), scale * np.sinh(s * t) / s
        if delta < -1e-12:  # complex pair -> spiral or centre
            w = np.sqrt(-delta)
            return scale * np.cos(w * t), scale * np.sin(w * t) / w
        # repeated eigenvalue -> degenerate node, the series stops after one term
        return scale, scale * t

    def _shifted(self) -> np.ndarray:
        return self.matrix - 0.5 * self.trace * np.eye(2)

    def flow(self, t: float) -> np.ndarray:
        """The propagator ``exp(M t)``: the map from day zero to time ``t``."""
        alpha, beta = self._propagator_coefficients(np.asarray(float(t)))
        return alpha * np.eye(2) + beta * self._shifted()

    def solve(self, state0, t) -> np.ndarray:
        """Exact states at the times ``t``.

        Returns an array of shape ``(len(t), 2)`` whose columns are Romeo's and
        Juliet's feelings.  Vectorised over ``t`` so that dragging a slider in
        the GUI re-solves the whole affair without a visible hitch.
        """
        state0 = np.asarray(state0, dtype=float)
        times = np.atleast_1d(np.asarray(t, dtype=float))
        alpha, beta = self._propagator_coefficients(times)
        rotated = self._shifted() @ state0
        return alpha[:, None] * state0 + beta[:, None] * rotated

    # -------------------------------------------------------------- analysis
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of ``M``, sorted by real part (descending)."""
        values = np.linalg.eigvals(self.matrix)
        return values[np.argsort(-values.real)]

    def eigenvectors(self) -> tuple[np.ndarray, np.ndarray]:
        """``(eigenvalues, eigenvectors)`` with eigenvectors in columns."""
        values, vectors = np.linalg.eig(self.matrix)
        order = np.argsort(-values.real)
        return values[order], vectors[:, order]

    def classify(self) -> Classification:
        """Name the fate of the relationship."""
        tr, det, disc = self.trace, self.determinant, self.discriminant
        tol = 1e-9

        if abs(det) < tol:
            if abs(tr) < tol and np.allclose(self.matrix, 0.0, atol=tol):
                return Classification(
                    "Frozen plane",
                    "neutral",
                    "Nothing moves. Whatever they felt on day one, they still feel.",
                )
            return Classification(
                "Line of fixed points",
                "unstable" if tr > tol else "stable" if tr < -tol else "neutral",
                "A whole line of possible endings: where they land depends entirely "
                "on where they started.",
            )

        if det < -tol:
            return Classification(
                "Saddle",
                "unstable",
                "A knife edge. One starting mood tips them into a love fest, the "
                "mirror-image mood tips them into a war.",
            )

        if disc < -tol:
            if abs(tr) < tol:
                return Classification(
                    "Centre",
                    "neutral",
                    "The Romeo & Juliet carousel: a closed orbit, forever. Love, "
                    "doubt, loathing, hope, repeat -- and never any closer to a "
                    "resolution.",
                )
            if tr < 0:
                return Classification(
                    "Stable spiral",
                    "stable",
                    "The storms get smaller each time round. They spiral into calm "
                    "mutual indifference.",
                )
            return Classification(
                "Unstable spiral",
                "unstable",
                "Every lap is louder than the last. The relationship oscillates its "
                "way off the chart.",
            )

        if abs(disc) < tol:
            return Classification(
                "Degenerate node",
                "stable" if tr < 0 else "unstable",
                "A single repeated eigenvalue: one shared direction, no arguing "
                "about which way to go.",
            )

        if tr < 0:
            return Classification(
                "Stable node",
                "stable",
                "No drama, no cycles. Both feelings decay straight to zero: they "
                "simply drift apart.",
            )
        return Classification(
            "Unstable node",
            "unstable",
            "Runaway feeling. They race off along an eigenvector -- mutual adoration "
            "or mutual hatred, and no way back.",
        )

    # ---------------------------------------------------------------- pretty
    def equations(self) -> tuple[str, str]:
        """The two ODEs as display strings, e.g. ``dx/dt = -y``."""

        def side(coef_x: float, coef_y: float) -> str:
            terms = []
            for coef, symbol in ((coef_x, "x"), (coef_y, "y")):
                if abs(coef) < 1e-9:
                    continue
                sign = "-" if coef < 0 else ("+" if terms else "")
                mag = abs(coef)
                magnitude = "" if abs(mag - 1.0) < 1e-9 else f"{mag:g}"
                terms.append(f"{sign} {magnitude}{symbol}".strip())
            return " ".join(terms) if terms else "0"

        return f"dx/dt = {side(self.a, self.b)}", f"dy/dt = {side(self.c, self.d)}"


def rk4_trajectory(affair: LoveAffair, state0, t_end: float, steps: int) -> np.ndarray:
    """Classic Runge-Kutta 4, kept for comparison with the exact solution.

    You never need this to run the model -- :meth:`LoveAffair.solve` is exact --
    but it is a fair demonstration of what a general-purpose ODE solver would do
    with the same system, and the test suite uses it to check the closed form.
    """
    state = np.asarray(state0, dtype=float)
    dt = t_end / steps
    out = np.empty((steps + 1, 2))
    out[0] = state
    for i in range(steps):
        k1 = affair.derivative(state)
        k2 = affair.derivative(state + 0.5 * dt * k1)
        k3 = affair.derivative(state + 0.5 * dt * k2)
        k4 = affair.derivative(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = state
    return out
