"""Command line front door: open the window, or render without one."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import LoveAffair
from .presets import DEFAULT, PRESETS, Preset, by_name

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="romeo-juliet",
        description="Romeo & Juliet as a 2x2 linear system: play with it, or render it.",
    )
    subcommands = parser.add_subparsers(dest="command")

    gui = subcommands.add_parser("gui", help="open the interactive window (default)")
    _add_affair_arguments(gui)

    render = subcommands.add_parser("render", help="write a PNG or GIF, no window")
    _add_affair_arguments(render)
    render.add_argument("-o", "--out", type=Path, default=Path("romeo-and-juliet.png"),
                        help="output image (.png/.svg/.pdf)")
    render.add_argument("--gif", type=Path, help="also write an animated GIF here")
    render.add_argument("--at", type=float, default=1.0,
                        help="fraction of the run to freeze the still on (0..1)")
    render.add_argument("--frames", type=int, default=120, help="frames in the GIF")
    render.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    render.add_argument("--gif-dpi", type=int, default=68,
                        help="resolution of the GIF (lower = smaller file)")

    listing = subcommands.add_parser("list", help="list the built-in relationships")
    listing.set_defaults(command="list")

    _add_affair_arguments(parser)
    return parser


def _add_affair_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", help="name of a built-in relationship")
    parser.add_argument("--params", type=float, nargs=4, metavar=("A", "B", "C", "D"),
                        help="the four entries of M, row by row")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X0", "Y0"),
                        help="how they feel on day one")
    parser.add_argument("--duration", type=float, default=24.0, help="length of the run")


def _resolve(args: argparse.Namespace) -> Preset:
    preset = by_name(args.preset) if args.preset else DEFAULT
    affair = LoveAffair(*args.params) if args.params else preset.affair
    start = tuple(args.start) if args.start else preset.start
    name = preset.name if not (args.params or args.start) else "Custom affair"
    story = preset.story if name == preset.name else "Parameters given on the command line."
    return Preset(name=name, affair=affair, start=start, story=story)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        for preset in PRESETS:
            affair = preset.affair
            print(f"{preset.name}")
            print(f"    M = [[{affair.a:g}, {affair.b:g}], [{affair.c:g}, {affair.d:g}]]"
                  f"   start = {preset.start}   -> {affair.classify().name}")
        return 0

    preset = _resolve(args)

    if args.command == "render":
        from .render import render_gif, render_still

        still = render_still(preset, args.out, at=args.at, duration=args.duration)
        print(f"wrote {still}")
        if args.gif:
            gif = render_gif(preset, args.gif, frames=args.frames, fps=args.fps,
                             duration=args.duration, dpi=args.gif_dpi)
            print(f"wrote {gif}")
        return 0

    from .gui import main as open_window

    open_window(preset)
    return 0
