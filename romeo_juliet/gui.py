"""The desktop app: sliders on the left, the couple's fate on the right."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .model import LoveAffair  # noqa: E402
from .plotting import LoveFigure, Simulation  # noqa: E402
from .presets import DEFAULT, PRESETS, Preset  # noqa: E402
from .theme import PALETTE, apply_rcparams  # noqa: E402

__all__ = ["LoveApp", "main"]

DURATION = 24.0
SAMPLES = 1400
FRAME_MS = 24

COEFFICIENTS = (
    ("a", "a  ·  Romeo ← his own feelings"),
    ("b", "b  ·  Romeo ← Juliet's feelings"),
    ("c", "c  ·  Juliet ← Romeo's feelings"),
    ("d", "d  ·  Juliet ← her own feelings"),
)


class LoveApp(ttk.Frame):
    """Tk front end for :class:`~romeo_juliet.model.LoveAffair`."""

    def __init__(self, master: tk.Misc, preset: Preset = DEFAULT, *,
                 animate: bool = True):
        super().__init__(master, padding=0)
        self.pack(fill="both", expand=True)

        apply_rcparams(matplotlib.rcParams)
        self._playing = True
        self._frame = 0
        self._pending: str | None = None
        self._timer: str | None = None
        self._loading = False

        self.vars = {key: tk.DoubleVar(value=getattr(preset.affair, key))
                     for key, _ in COEFFICIENTS}
        self.vars["x0"] = tk.DoubleVar(value=preset.start[0])
        self.vars["y0"] = tk.DoubleVar(value=preset.start[1])
        self.speed = tk.DoubleVar(value=3.0)
        self.show_field = tk.BooleanVar(value=True)
        self.show_eigen = tk.BooleanVar(value=True)
        self.preset_name = tk.StringVar(value=preset.name)
        self.story = tk.StringVar(value=preset.story)

        self._build_canvas()
        self._build_controls()

        for var in self.vars.values():
            var.trace_add("write", self._on_parameter_change)
        self.show_field.trace_add("write", self._on_view_change)
        self.show_eigen.trace_add("write", self._on_view_change)

        self._rebuild(reset_frame=True)
        if animate:
            self._tick()

        master.bind("<space>", lambda _e: self.toggle_play())
        master.bind("<r>", lambda _e: self.restart())
        master.bind("<Escape>", lambda _e: master.destroy())

    # ------------------------------------------------------------------ build
    def _build_canvas(self) -> None:
        holder = ttk.Frame(self)
        holder.pack(side="right", fill="both", expand=True)
        self.figure = Figure(figsize=(8.2, 5.7), dpi=100)
        self.view = LoveFigure(self.figure)
        self.canvas = FigureCanvasTkAgg(self.figure, master=holder)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _scrolling_panel(self, width: int = 352) -> ttk.Frame:
        """A fixed-width column that grows a scrollbar only when it needs one.

        Preset descriptions differ in length and laptops differ in height, so
        the controls cannot be assumed to fit; this way they always reach.
        """
        outer = ttk.Frame(self)
        outer.pack(side="left", fill="y")
        canvas = tk.Canvas(outer, width=width, highlightthickness=0,
                           background=PALETTE["bg"])
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="y")

        inner = ttk.Frame(canvas, padding=(14, 12))
        window = canvas.create_window((0, 0), window=inner, anchor="nw")

        def fit(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window, width=canvas.winfo_width())
            overflowing = inner.winfo_reqheight() > canvas.winfo_height()
            if overflowing and not bar.winfo_ismapped():
                bar.pack(side="right", fill="y")
            elif not overflowing and bar.winfo_ismapped():
                bar.pack_forget()

        inner.bind("<Configure>", fit)
        canvas.bind("<Configure>", fit)

        def wheel(event) -> None:
            widget = self.winfo_containing(event.x_root, event.y_root)
            while widget is not None:  # only scroll when the pointer is over us
                if widget in (canvas, inner):
                    canvas.yview_scroll(-event.delta // 120, "units")
                    return
                widget = widget.master

        self.bind_all("<MouseWheel>", wheel, add="+")
        return inner

    def _build_controls(self) -> None:
        panel = self._scrolling_panel()

        ttk.Label(panel, text="Romeo & Juliet", font=("Segoe UI", 15, "bold")).pack(
            anchor="w"
        )
        ttk.Label(panel, text="a linear model of two people's feelings",
                  foreground=PALETTE["muted"]).pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="Relationship", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        picker = ttk.Combobox(panel, textvariable=self.preset_name, state="readonly",
                              values=[p.name for p in PRESETS], width=34)
        picker.pack(anchor="w", fill="x", pady=(2, 6))
        picker.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Label(panel, textvariable=self.story, wraplength=310, justify="left",
                  foreground=PALETTE["muted"]).pack(anchor="w", pady=(0, 8))

        ttk.Separator(panel).pack(fill="x", pady=3)
        ttk.Label(panel, text="The matrix M", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        for key, caption in COEFFICIENTS:
            self._slider(panel, caption, self.vars[key], -3.0, 3.0)

        ttk.Separator(panel).pack(fill="x", pady=3)
        ttk.Label(panel, text="Where they start", font=("Segoe UI", 9, "bold")).pack(
            anchor="w"
        )
        self._slider(panel, "x₀  ·  Romeo on day one", self.vars["x0"], -2.0, 2.0)
        self._slider(panel, "y₀  ·  Juliet on day one", self.vars["y0"], -2.0, 2.0)
        self._slider(panel, "speed", self.speed, 0.0, 10.0)

        ttk.Separator(panel).pack(fill="x", pady=3)
        toggles = ttk.Frame(panel)
        toggles.pack(anchor="w", fill="x")
        ttk.Checkbutton(toggles, text="vector field", variable=self.show_field).pack(
            side="left"
        )
        ttk.Checkbutton(toggles, text="eigendirections", variable=self.show_eigen).pack(
            side="left", padx=(12, 0)
        )

        buttons = ttk.Frame(panel)
        buttons.pack(anchor="w", fill="x", pady=(8, 6))
        self.play_button = ttk.Button(buttons, text="Pause", width=8, command=self.toggle_play)
        self.play_button.pack(side="left")
        for text, command in (("Restart", self.restart), ("Random", self.randomise),
                              ("Save", self.save_image)):
            ttk.Button(buttons, text=text, width=8, command=command).pack(
                side="left", padx=(3, 0)
            )

        ttk.Separator(panel).pack(fill="x", pady=3)
        self.verdict = ttk.Label(panel, text="", font=("Segoe UI", 10, "bold"),
                                 wraplength=310, justify="left")
        self.verdict.pack(anchor="w")
        self.detail = ttk.Label(panel, text="", wraplength=310, justify="left",
                                foreground=PALETTE["muted"])
        self.detail.pack(anchor="w", pady=(2, 0))
        self.numbers = ttk.Label(panel, text="", wraplength=310, justify="left",
                                 font=("Consolas", 8), foreground=PALETTE["muted"])
        self.numbers.pack(anchor="w", pady=(6, 0))
        ttk.Label(panel, text="space = play/pause    r = restart    esc = quit",
                  foreground=PALETTE["muted"], font=("Segoe UI", 8)).pack(
            anchor="w", pady=(10, 0)
        )

    def _slider(self, parent, caption, variable, low, high) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(2, 0))
        head = ttk.Frame(row)
        head.pack(fill="x")
        ttk.Label(head, text=caption, font=("Segoe UI", 8)).pack(side="left")
        readout = ttk.Label(head, text=f"{variable.get():+.2f}", font=("Consolas", 8),
                            foreground=PALETTE["ink"])
        readout.pack(side="right")
        scale = ttk.Scale(row, from_=low, to=high, variable=variable, orient="horizontal")
        scale.pack(fill="x")

        def refresh(*_args, _label=readout, _var=variable) -> None:
            _label.configure(text=f"{_var.get():+.2f}")

        # Only the variables in ``self.vars`` trigger a re-solve; ``speed`` is
        # wired up the same way here but changes nothing but the frame rate.
        variable.trace_add("write", refresh)

    # --------------------------------------------------------------- reactions
    def _on_parameter_change(self, *_args) -> None:
        if self._loading:
            return
        self.preset_name.set("custom")
        self.story.set("Your own affair. Drag the sliders and watch the orbit change.")
        self._schedule_rebuild(reset_frame=True)

    def _on_view_change(self, *_args) -> None:
        self.view.show_field = self.show_field.get()
        self.view.show_eigen = self.show_eigen.get()
        self._schedule_rebuild(reset_frame=False)

    def _on_preset_selected(self, _event=None) -> None:
        try:
            preset = next(p for p in PRESETS if p.name == self.preset_name.get())
        except StopIteration:
            return
        self.load(preset)

    def load(self, preset: Preset) -> None:
        """Push a preset into the sliders without treating it as a user edit."""
        self._loading = True
        try:
            for key, _ in COEFFICIENTS:
                self.vars[key].set(getattr(preset.affair, key))
            self.vars["x0"].set(preset.start[0])
            self.vars["y0"].set(preset.start[1])
            self.preset_name.set(preset.name)
            self.story.set(preset.story)
        finally:
            self._loading = False
        self._rebuild(reset_frame=True)

    def randomise(self) -> None:
        rng = np.random.default_rng()
        self._loading = True
        try:
            for key, _ in COEFFICIENTS:
                self.vars[key].set(round(float(rng.uniform(-2.0, 2.0)), 2))
            self.vars["x0"].set(round(float(rng.uniform(-1.5, 1.5)), 2))
            self.vars["y0"].set(round(float(rng.uniform(-1.5, 1.5)), 2))
            self.preset_name.set("custom")
            self.story.set("A relationship drawn out of a hat.")
        finally:
            self._loading = False
        self._rebuild(reset_frame=True)

    def toggle_play(self) -> None:
        self._playing = not self._playing
        self.play_button.configure(text="Pause" if self._playing else "Play")

    def restart(self) -> None:
        self._frame = 0
        self.view.set_frame(0)
        self.canvas.draw_idle()

    def save_image(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("SVG image", "*.svg"), ("PDF", "*.pdf")],
            initialfile="romeo-and-juliet.png",
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=200, facecolor=self.figure.get_facecolor())
        except OSError as error:
            messagebox.showerror("Could not save", str(error))
        else:
            messagebox.showinfo("Saved", f"Written to\n{path}")

    # ------------------------------------------------------------------ engine
    def _schedule_rebuild(self, *, reset_frame: bool) -> None:
        """Coalesce a burst of slider events into a single redraw."""
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(20, lambda: self._rebuild(reset_frame=reset_frame))

    def _rebuild(self, *, reset_frame: bool) -> None:
        self._pending = None
        affair = LoveAffair(*(self.vars[key].get() for key, _ in COEFFICIENTS))
        start = (self.vars["x0"].get(), self.vars["y0"].get())
        simulation = Simulation.run(affair, start, duration=DURATION, samples=SAMPLES)
        self.view.set_simulation(simulation)

        if reset_frame:
            self._frame = 0
        self._frame = min(self._frame, simulation.size - 1)
        self.view.set_frame(self._frame)
        self._describe(affair)
        self.canvas.draw_idle()

    def _describe(self, affair: LoveAffair) -> None:
        verdict = affair.classify()
        values = affair.eigenvalues()
        self.verdict.configure(text=f"{verdict.name} — {verdict.stability}")
        self.detail.configure(text=verdict.blurb)
        self.numbers.configure(
            text=(
                f"tr M = {affair.trace:+.2f}   det M = {affair.determinant:+.2f}\n"
                f"λ₁ = {_format_complex(values[0])}\n"
                f"λ₂ = {_format_complex(values[1])}"
            )
        )

    def _advance(self) -> bool:
        """Move one animation step. Returns whether anything actually moved."""
        simulation = self.view.simulation
        if not self._playing or simulation is None or simulation.size <= 1:
            return False
        step = max(int(round(self.speed.get())), 0)
        if not step:
            return False
        self._frame = (self._frame + step) % simulation.size
        self.view.set_frame(self._frame)
        self.canvas.draw_idle()
        return True

    def _tick(self) -> None:
        self._advance()
        self._timer = self.after(FRAME_MS, self._tick)

    def destroy(self) -> None:  # pragma: no cover - teardown path
        for handle in (self._timer, self._pending):
            if handle is not None:
                try:
                    self.after_cancel(handle)
                except tk.TclError:
                    pass
        self._timer = self._pending = None
        super().destroy()


def _format_complex(value: complex) -> str:
    if abs(value.imag) < 1e-9:
        return f"{value.real:+.3f}"
    return f"{value.real:+.3f} {'+' if value.imag >= 0 else '-'} {abs(value.imag):.3f}i"


def main(preset: Preset = DEFAULT) -> None:
    """Open the window."""
    root = tk.Tk()
    root.title("Romeo & Juliet — a linear love affair")

    # Fit the screen we actually have, centred, rather than a fixed size that
    # hangs off the edge of a laptop display.
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    width, height = min(1220, int(screen_w * 0.92)), min(770, int(screen_h * 0.90))
    root.geometry(f"{width}x{height}+{(screen_w - width) // 2}+{(screen_h - height) // 3}")
    root.minsize(min(940, width), min(600, height))
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:  # pragma: no cover - platform dependent
        pass
    style = ttk.Style()
    style.configure("TFrame", background=PALETTE["bg"])
    style.configure("TLabel", background=PALETTE["bg"], foreground=PALETTE["ink"])
    style.configure("TCheckbutton", background=PALETTE["bg"], foreground=PALETTE["ink"])
    style.configure("TSeparator", background=PALETTE["grid"])
    root.configure(background=PALETTE["bg"])

    LoveApp(root, preset)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    main()
