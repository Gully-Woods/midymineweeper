"""
Core Minesweeper board logic.

No pygame imports here — this module is used by both the human-playable
pygame frontend and the headless RL environment.

Cell view states stored in self.view (numpy int array):
  -1  HIDDEN
   0  REVEALED, 0 adjacent mines (blank)
  1-8 REVEALED, N adjacent mines
   9  FLAGGED
"""

import numpy as np

HIDDEN  = -1
FLAGGED =  9


class Board:
    def __init__(self, rows: int, cols: int, num_mines: int, seed: int = None):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self._rng = np.random.default_rng(seed)

        # Mines and adjacency numbers are unknown until first reveal
        self._mines: np.ndarray = None   # bool (rows, cols)
        self._numbers: np.ndarray = None # int  (rows, cols), 0-8

        # What the player (and RL agent) can see
        self.view = np.full((rows, cols), HIDDEN, dtype=np.int8)
        self._mines_placed = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def place_mines(self, safe_cell: tuple):
        """Place mines randomly, guaranteeing safe_cell is mine-free."""
        row, col = safe_cell
        all_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)
                     if (r, c) != (row, col)]
        chosen = self._rng.choice(len(all_cells), size=self.num_mines, replace=False)
        self._mines = np.zeros((self.rows, self.cols), dtype=bool)
        for idx in chosen:
            r, c = all_cells[idx]
            self._mines[r, c] = True
        self._compute_numbers()
        self._mines_placed = True

    def _compute_numbers(self):
        """Precompute adjacent mine counts for every cell."""
        self._numbers = np.zeros((self.rows, self.cols), dtype=np.int8)
        for r in range(self.rows):
            for c in range(self.cols):
                if self._mines[r, c]:
                    continue
                self._numbers[r, c] = self._mines[
                    max(0, r-1):r+2, max(0, c-1):c+2
                ].sum()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def reveal(self, row: int, col: int) -> str:
        """
        Reveal a cell. Returns:
          "mine"             — hit a mine
          "safe"             — revealed at least one cell
          "already_revealed" — cell was already visible
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return "already_revealed"
        if self.view[row, col] != HIDDEN:
            return "already_revealed"

        if not self._mines_placed:
            self.place_mines((row, col))

        if self._mines[row, col]:
            self.view[row, col] = self._numbers[row, col]  # reveal for death screen
            return "mine"

        self._flood_fill(row, col)
        return "safe"

    def _flood_fill(self, row: int, col: int):
        """Recursively reveal blank (0-adjacent) regions."""
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                continue
            if self.view[r, c] != HIDDEN:
                continue
            self.view[r, c] = self._numbers[r, c]
            if self._numbers[r, c] == 0:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        stack.append((r + dr, c + dc))

    def flag(self, row: int, col: int):
        """Toggle flag on a hidden cell."""
        if self.view[row, col] == HIDDEN:
            self.view[row, col] = FLAGGED
        elif self.view[row, col] == FLAGGED:
            self.view[row, col] = HIDDEN

    def reveal_all_mines(self):
        """Expose all mines — called on game over for display."""
        if self._mines is not None:
            for r in range(self.rows):
                for c in range(self.cols):
                    if self._mines[r, c]:
                        self.view[r, c] = FLAGGED  # reuse flag slot to mark mines

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_won(self) -> bool:
        """Win = every non-mine cell is revealed."""
        if not self._mines_placed:
            return False
        for r in range(self.rows):
            for c in range(self.cols):
                if not self._mines[r, c] and self.view[r, c] == HIDDEN:
                    return False
        return True

    def get_observation(self) -> np.ndarray:
        """Flat int8 array of shape (rows*cols,) for the RL agent."""
        return self.view.flatten().copy()

    def hidden_cells(self) -> list:
        """Flat indices of all currently hidden cells."""
        flat = self.view.flatten()
        return [i for i, v in enumerate(flat) if v == HIDDEN]

    def mines_remaining(self) -> int:
        flags = int((self.view == FLAGGED).sum())
        return self.num_mines - flags
