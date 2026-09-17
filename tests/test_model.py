import numpy as np
import pytest

from romeo_juliet import PRESETS, LoveAffair, Simulation, by_name, rk4_trajectory

CLASSIC = LoveAffair(a=0.0, b=-1.0, c=1.0, d=0.0)


def test_classic_solution_is_cos_and_sin():
    """dx/dt = -y, dy/dt = x with (1, 0) is exactly (cos t, sin t)."""
    times = np.linspace(0.0, 4.0 * np.pi, 97)
    states = CLASSIC.solve([1.0, 0.0], times)
    assert np.allclose(states[:, 0], np.cos(times), atol=1e-12)
    assert np.allclose(states[:, 1], np.sin(times), atol=1e-12)


def test_classic_orbit_is_a_circle():
    times = np.linspace(0.0, 20.0, 500)
    radii = np.linalg.norm(CLASSIC.solve([0.6, -0.8], times), axis=1)
    assert np.allclose(radii, 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "affair",
    [
        CLASSIC,
        LoveAffair(-1.0, 2.0, 2.0, -1.0),  # saddle, distinct real eigenvalues
        LoveAffair(1.0, 1.0, 1.0, 1.0),  # singular, zero eigenvalue
        LoveAffair(2.0, 1.0, 0.0, 2.0),  # repeated eigenvalue, not diagonalisable
        LoveAffair(-0.4, -1.5, 1.5, -0.4),  # stable spiral
    ],
)
def test_closed_form_matches_runge_kutta(affair):
    """The analytic propagator agrees with a general-purpose integrator."""
    start = np.array([0.7, -0.3])
    t_end, steps = 3.0, 6000
    numeric = rk4_trajectory(affair, start, t_end, steps)[-1]
    exact = affair.solve(start, [t_end])[0]
    assert np.allclose(numeric, exact, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("affair", [CLASSIC, LoveAffair(0.3, -1.2, 0.9, -0.5),
                                    LoveAffair(1.0, 0.0, 0.0, 1.0)])
def test_flow_is_a_group(affair):
    """exp(M(s+t)) == exp(Ms) exp(Mt), and exp(0) == I."""
    assert np.allclose(affair.flow(0.0), np.eye(2))
    assert np.allclose(affair.flow(1.7) @ affair.flow(0.9), affair.flow(2.6))


def test_solve_satisfies_the_differential_equation():
    affair = LoveAffair(0.5, -1.3, 0.8, -0.2)
    start, t, h = np.array([1.0, 0.4]), 1.1, 1e-6
    ahead, behind = affair.solve(start, [t + h, t - h])
    assert np.allclose((ahead - behind) / (2 * h), affair.derivative(affair.solve(start, [t])[0]),
                       atol=1e-6)


@pytest.mark.parametrize(
    "affair, expected",
    [
        (LoveAffair(0, -1, 1, 0), "Centre"),
        (LoveAffair(-1, 2, 2, -1), "Saddle"),
        (LoveAffair(-0.4, -1.5, 1.5, -0.4), "Stable spiral"),
        (LoveAffair(0.25, -1.5, 1.5, 0.25), "Unstable spiral"),
        (LoveAffair(-2, 0, 0, -1), "Stable node"),
        (LoveAffair(2, 0, 0, 1), "Unstable node"),
        (LoveAffair(-2, 1, 0, -2), "Degenerate node"),
        (LoveAffair(1, 1, 1, 1), "Line of fixed points"),
        (LoveAffair(0, 0, 0, 0), "Frozen plane"),
    ],
)
def test_classification(affair, expected):
    assert affair.classify().name == expected


def test_eigenvalues_are_sorted_by_real_part():
    values = LoveAffair(-2.0, 0.0, 0.0, 3.0).eigenvalues()
    assert values[0].real > values[1].real


def test_equations_render_readably():
    assert CLASSIC.equations() == ("dx/dt = - y", "dy/dt = x")
    assert LoveAffair(2, 0, 0, 0).equations() == ("dx/dt = 2x", "dy/dt = 0")


def test_presets_have_unique_names_and_are_reachable():
    names = [preset.name for preset in PRESETS]
    assert len(names) == len(set(names))
    for name in names:
        assert by_name(name.upper()).name == name
    with pytest.raises(KeyError):
        by_name("no such couple")


def test_simulation_trims_runaway_trajectories():
    """An explosive affair is followed only as far as it is worth plotting."""
    runaway = Simulation.run(LoveAffair(3.0, 0.0, 0.0, 3.0), (1.0, 1.0), duration=40.0)
    assert runaway.size < 1400
    assert np.isfinite(runaway.states).all()
    assert np.linalg.norm(runaway.states, axis=1).max() < 1e4


def test_simulation_keeps_bounded_trajectories_whole():
    calm = Simulation.run(CLASSIC, (1.0, 0.0), duration=24.0, samples=500)
    assert calm.size == 500
    assert calm.limits() > 1.0
