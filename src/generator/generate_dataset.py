"""
Generate a dataset of no-guess Minesweeper boards and save to data/.

Usage:
    python -m src.generator.generate_dataset --preset beginner --train 5000 --test 1000
    python -m src.generator.generate_dataset --preset intermediate --train 2000 --test 500

Output files:
    data/<preset>_train.npz  —  mines array of shape (N, rows, cols), dtype bool
    data/<preset>_test.npz   —  same format
"""

import argparse
import os
import numpy as np

from .no_guess_board import generate_no_guess_board
from ..config.settings import PRESETS


def generate_split(preset: str, n: int, start_cell: tuple, seed: int) -> np.ndarray:
    cfg = PRESETS[preset]
    rows, cols, num_mines = cfg["rows"], cfg["cols"], cfg["num_mines"]
    rng = np.random.default_rng(seed)
    boards = np.zeros((n, rows, cols), dtype=bool)
    for i in range(n):
        mines, _ = generate_no_guess_board(rows, cols, num_mines, start_cell, rng)
        boards[i] = mines
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  {i + 1}/{n} boards generated")
    return boards


def main():
    parser = argparse.ArgumentParser(description="Generate no-guess Minesweeper board datasets")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="beginner")
    parser.add_argument("--train", type=int, default=5000, help="Number of training boards")
    parser.add_argument("--test",  type=int, default=1000, help="Number of test boards")
    parser.add_argument("--seed",  type=int, default=42,   help="Base random seed")
    parser.add_argument("--out",   type=str, default="data", help="Output directory")
    args = parser.parse_args()

    cfg = PRESETS[args.preset]
    rows, cols = cfg["rows"], cfg["cols"]
    start_cell = (rows // 2, cols // 2)

    os.makedirs(args.out, exist_ok=True)

    print(f"Preset: {args.preset}  ({rows}x{cols}, {cfg['num_mines']} mines)")
    print(f"Start cell: {start_cell}")

    print(f"\nGenerating {args.train} training boards...")
    train_boards = generate_split(args.preset, args.train, start_cell, seed=args.seed)
    train_path = os.path.join(args.out, f"{args.preset}_train.npz")
    np.savez_compressed(train_path, mines=train_boards)
    print(f"Saved: {train_path}  ({train_boards.nbytes // 1024} KB uncompressed)")

    print(f"\nGenerating {args.test} test boards...")
    test_boards = generate_split(args.preset, args.test, start_cell, seed=args.seed + 1_000_000)
    test_path = os.path.join(args.out, f"{args.preset}_test.npz")
    np.savez_compressed(test_path, mines=test_boards)
    print(f"Saved: {test_path}  ({test_boards.nbytes // 1024} KB uncompressed)")

    print(f"\nDone. {args.train + args.test} boards total.")


if __name__ == "__main__":
    main()
