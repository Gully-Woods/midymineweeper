"""
Constraint-propagation solver for Minesweeper.

Used by the no-guess board generator to test whether a given mine layout
is solvable without guessing, starting from a given cell.

Never mutates game state — works on the mines/numbers arrays directly.
"""

import numpy as np
from collections import deque


def is_no_guess(
    mines: np.ndarray,    # bool (rows, cols)
    numbers: np.ndarray,  # int8 (rows, cols)
    start: tuple,         # (row, col) — first cell revealed; must not be a mine
) -> bool:
    """
    Return True if the board is solvable by single-point constraint
    propagation alone, starting from `start`.
    """
    rows, cols = mines.shape
    revealed: set = set()
    flagged: set = set()

    def neighbors(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

    def flood_fill(r, c):
        queue = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            if (cr, cc) in revealed or mines[cr, cc]:
                continue
            revealed.add((cr, cc))
            if numbers[cr, cc] == 0:
                for nr, nc in neighbors(cr, cc):
                    if (nr, nc) not in revealed:
                        queue.append((nr, nc))

    flood_fill(*start)

    changed = True
    while changed:
        changed = False
        for r, c in list(revealed):
            if numbers[r, c] == 0:
                continue

            hidden_nbrs = [
                (nr, nc) for nr, nc in neighbors(r, c)
                if (nr, nc) not in revealed and (nr, nc) not in flagged
            ]
            flagged_count = sum(1 for nb in neighbors(r, c) if nb in flagged)

            # Rule 1: all remaining hidden neighbors must be mines
            if numbers[r, c] == flagged_count + len(hidden_nbrs) and hidden_nbrs:
                for cell in hidden_nbrs:
                    flagged.add(cell)
                changed = True

            # Rule 2: all mines accounted for — hidden neighbors are safe
            if numbers[r, c] == flagged_count and hidden_nbrs:
                for nr, nc in hidden_nbrs:
                    flood_fill(nr, nc)
                changed = True

    total_safe = int((~mines).sum())
    return len(revealed) == total_safe
