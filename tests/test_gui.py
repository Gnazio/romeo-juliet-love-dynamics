"""Smoke tests for the window: it builds, it reacts, it tears down.

Skipped automatically wherever Tk cannot open a display (most CI runners).
"""

import pytest

tkinter = pytest.importorskip("tkinter")


@pytest.fixture
def root():
    try:
        window = tkinter.Tk()
    except tkinter.TclError as error:  # pragma: no cover - headless machines
        pytest.skip(f"no display available: {error}")
    window.withdraw()
    try:
        yield window
    finally:
        window.destroy()


def test_app_builds_and_reacts(root):
    from romeo_juliet.gui import LoveApp
    from romeo_juliet.presets import PRESETS

    app = LoveApp(root, animate=False)
    root.update()
    assert app.view.simulation is not None
    assert app.verdict.cget("text").startswith("Centre")

    app.load(PRESETS[1])  # a saddle
    root.update()
    assert app.view.simulation.affair == PRESETS[1].affair
    assert "Saddle" in app.verdict.cget("text")

    app.vars["b"].set(0.5)  # editing a slider makes the affair "custom"
    app._rebuild(reset_frame=True)
    root.update()
    assert app.preset_name.get() == "custom"
    assert app.view.simulation.affair.b == pytest.approx(0.5)


def test_playback_advances_and_pauses(root):
    from romeo_juliet.gui import LoveApp

    app = LoveApp(root, animate=False)
    root.update()
    app.speed.set(5.0)
    for _ in range(4):
        app._advance()
        root.update()
    assert app._frame > 0

    app.toggle_play()
    frozen = app._frame
    for _ in range(4):
        app._advance()
        root.update()
    assert app._frame == frozen
    assert app.play_button.cget("text") == "Play"

    app.restart()
    assert app._frame == 0


def test_randomise_keeps_the_figure_valid(root):
    from romeo_juliet.gui import LoveApp

    app = LoveApp(root, animate=False)
    for _ in range(5):
        app.randomise()
        root.update()
        assert app.view.simulation.size >= 10
