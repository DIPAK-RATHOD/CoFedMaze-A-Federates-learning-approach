"""
main.py

CLI utility for generating, validating, and rendering a single maze.

Deliberately NOT a training entry point — per project convention, agent
training requires the PettingZoo wrapper and VDN training loop (neither
built yet), and lives in scripts/train_local.py / train_federated.py
once those exist. This file only ever imports from env/, never from
marl/, to keep that boundary structural rather than just a convention
someone has to remember.

Usage (from the repository root):
    python main.py
    python main.py --rows 15 --cols 21 --algorithm kruskal --seed 42
    python main.py --validate --render both
    python main.py --list-algorithms
"""

import argparse
import os
import random
import sys
from typing import List, Optional

from env.core.maze import Maze
from env.generator.generator_factory import available_generators, create_generator
from env.render.ascii_renderer import AsciiRenderer
from env.render.matplotlib_renderer import MatplotlibRenderer
from env.validators.connectivity import is_fully_connected
from env.validators.path_checker import has_path_to_exit, shortest_path_length

DEFAULT_ROWS = 15
DEFAULT_COLS = 21
DEFAULT_OUTPUT_DIR = os.path.join("data", "generated")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate, validate, and render a CoFedMaze maze.",
        epilog="Example: python main.py --rows 15 --cols 21 --algorithm kruskal --seed 42",
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS,
        help=f"Raw grid rows, must be odd (default: {DEFAULT_ROWS}).",
    )
    parser.add_argument(
        "--cols", type=int, default=DEFAULT_COLS,
        help=f"Raw grid cols, must be odd (default: {DEFAULT_COLS}).",
    )
    parser.add_argument(
        "--algorithm", choices=available_generators(), default="recursive_backtracking",
        help="Maze generation algorithm (default: recursive_backtracking).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed. If omitted, a random seed is chosen and printed "
        "so the run can be reproduced later.",
    )
    parser.add_argument(
        "--render", choices=["ascii", "matplotlib", "both", "none"], default="ascii",
        help="Which renderer(s) to run (default: ascii).",
    )
    parser.add_argument(
        "--save-path", type=str, default=None,
        help=f"Output path for the matplotlib render. Defaults to "
        f"{DEFAULT_OUTPUT_DIR}/maze_seed_<seed>.png.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run connectivity and start-to-exit solvability checks and print the result.",
    )
    parser.add_argument(
        "--list-algorithms", action="store_true",
        help="Print the available generation algorithms and exit.",
    )
    return parser


def resolve_seed(requested_seed: Optional[int]) -> int:
    """
    Return `requested_seed` if given, else a freshly chosen random seed.

    A random-but-recorded seed (rather than leaving Maze/generator to
    default to unreproducible OS entropy) means every run this CLI
    produces can be reproduced later just by passing --seed with the
    printed value — important given how much this project's
    reproducibility story already depends on seeds (env/utils/random_seed.py,
    data/nodeN/seeds/*.json).
    """
    if requested_seed is not None:
        return requested_seed
    return random.randint(0, 2**31 - 1)


def default_save_path(seed: int) -> str:
    """
    Default matplotlib output path, matching the existing
    data/generated/maze_seed_<seed>.png naming convention already used
    in this repository.
    """
    return os.path.join(DEFAULT_OUTPUT_DIR, f"maze_seed_{seed}.png")


def run(argv: Optional[List[str]] = None) -> int:
    """
    Parse arguments and execute the requested generate/validate/render
    pipeline. Returns a process exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_algorithms:
        print("Available algorithms:", ", ".join(available_generators()))
        return 0

    seed = resolve_seed(args.seed)

    try:
        maze = Maze(rows=args.rows, cols=args.cols, random_seed=seed)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    generator = create_generator(args.algorithm, random_seed=seed)
    generator.generate(maze)

    print(f"Generated {maze.rows}x{maze.cols} maze with '{args.algorithm}' (seed={seed})")
    print(f"Start (logical): {maze.start_position}   Exit (logical): {maze.exit_position}")

    if args.validate:
        connected = is_fully_connected(maze)
        solvable = has_path_to_exit(maze)
        print(f"Fully connected: {connected}")
        print(f"Has path to exit: {solvable}")
        if solvable:
            print(f"Shortest path length (raw-grid moves): {shortest_path_length(maze)}")

    if args.render in ("ascii", "both"):
        print()
        AsciiRenderer().print_maze(maze)

    if args.render in ("matplotlib", "both"):
        save_path = args.save_path or default_save_path(seed)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        MatplotlibRenderer().render(maze, save_path=save_path)
        print(f"Saved rendering to {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
