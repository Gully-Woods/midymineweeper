"""
MidyMineweeper — entry point.

Usage:
    python main.py                    # play with beginner board
    python main.py --preset intermediate
    python main.py --preset expert
    python main.py --seed 42          # reproducible board
"""

import argparse
from src.game.game import Game
from src.game.renderer import Renderer
from src.config.settings import PRESETS, DEFAULT_PRESET


def main():
    parser = argparse.ArgumentParser(description="MidyMineweeper")
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=DEFAULT_PRESET,
        help="Board difficulty preset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible boards",
    )
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    game = Game(cfg["rows"], cfg["cols"], cfg["num_mines"], seed=args.seed)
    renderer = Renderer(game)
    renderer.run()


if __name__ == "__main__":
    main()
