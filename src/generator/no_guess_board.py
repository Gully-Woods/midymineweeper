"""
No-guess board generator using rejection sampling.

Repeatedly generates random mine layouts until the constraint solver
confirms the board is solvable without guessing from the start cell.
"""

import numpy as np
from ..solver.constraint_solver import is_no_guess

_DEFAULT_MAX_ATTEMPTS = 10_000


def generate_no_guess_board(
    rows: int,
    cols: int,
    num_mines: int,
    start_cell: tuple,
    rng: np.random.Generator,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
) -> tuple:
    """
    Generate a mine layout solvable by constraint propagation from `start_cell`.

    Parameters
    ----------
    rows, cols   : board dimensions
    num_mines    : number of mines to place
    start_cell   : (row, col) that will be the first reveal — excluded from mines
                   and used as the solver's starting point
    rng          : caller-owned numpy Generator (preserves seed reproducibility)
    max_attempts : raise RuntimeError after this many failed attempts

    Returns
    -------
    mines   : bool ndarray of shape (rows, cols)
    numbers : int8 ndarray of shape (rows, cols)
    """
    all_cells = [
        (r, c) for r in range(rows) for c in range(cols)
        if (r, c) != start_cell
    ]
    n = len(all_cells)

    for _ in range(max_attempts):
        chosen_idx = rng.choice(n, size=num_mines, replace=False)
        mines = np.zeros((rows, cols), dtype=bool)
        for idx in chosen_idx:
            r, c = all_cells[idx]
            mines[r, c] = True

        numbers = _compute_numbers(mines, rows, cols)

        if is_no_guess(mines, numbers, start_cell):
            return mines, numbers

    raise RuntimeError(
        f"No no-guess board found after {max_attempts} attempts "
        f"({rows}x{cols}, {num_mines} mines). "
        "Consider reducing mine density or increasing max_attempts."
    )


def _compute_numbers(mines: np.ndarray, rows: int, cols: int) -> np.ndarray:
    numbers = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            if mines[r, c]:
                continue
            numbers[r, c] = mines[max(0, r-1):r+2, max(0, c-1):c+2].sum()
    return numbers
