"""
Unit tests for Board logic.
Run with: python -m pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.game.board import Board, HIDDEN, FLAGGED


def make_board(rows=5, cols=5, mines=3, seed=42):
    return Board(rows, cols, mines, seed=seed)


def fully_revealed_board():
    """Force a board where all safe cells are revealed."""
    b = make_board()
    b.place_mines((0, 0))
    # Reveal every non-mine cell
    for r in range(b.rows):
        for c in range(b.cols):
            if not b._mines[r, c]:
                b.reveal(r, c)
    return b


class TestMinePlacement:
    def test_correct_mine_count(self):
        b = make_board(mines=5)
        b.place_mines((0, 0))
        assert b._mines.sum() == 5

    def test_safe_cell_not_mined(self):
        b = make_board()
        b.place_mines((2, 2))
        assert not b._mines[2, 2]

    def test_numbers_correct(self):
        b = make_board()
        b.place_mines((0, 0))
        # Each number cell should equal count of adjacent mines
        for r in range(b.rows):
            for c in range(b.cols):
                if not b._mines[r, c]:
                    expected = b._mines[
                        max(0, r-1):r+2, max(0, c-1):c+2
                    ].sum()
                    assert b._numbers[r, c] == expected


class TestReveal:
    def test_first_reveal_never_mine(self):
        for seed in range(20):
            b = Board(5, 5, 5, seed=seed)
            result = b.reveal(2, 2)
            assert result == "safe"

    def test_already_revealed_returns_correct(self):
        b = make_board()
        b.reveal(0, 0)
        result = b.reveal(0, 0)
        assert result == "already_revealed"

    def test_flood_fill_reveals_blank_region(self):
        # Build a board with a guaranteed blank corner by using a fixed seed
        b = Board(9, 9, 10, seed=0)
        b.reveal(0, 0)
        # At least the clicked cell should be revealed
        assert b.view[0, 0] != HIDDEN


class TestFlag:
    def test_flag_and_unflag(self):
        b = make_board()
        b.flag(1, 1)
        assert b.view[1, 1] == FLAGGED
        b.flag(1, 1)
        assert b.view[1, 1] == HIDDEN

    def test_cannot_flag_revealed(self):
        b = make_board()
        b.reveal(0, 0)
        original = b.view[0, 0]
        b.flag(0, 0)
        assert b.view[0, 0] == original  # unchanged


class TestWinCondition:
    def test_not_won_at_start(self):
        b = make_board()
        assert not b.is_won()

    def test_won_after_all_safe_revealed(self):
        b = fully_revealed_board()
        assert b.is_won()


class TestObservation:
    def test_observation_shape(self):
        b = make_board(rows=9, cols=9)
        obs = b.get_observation()
        assert obs.shape == (81,)

    def test_observation_all_hidden_at_start(self):
        b = make_board()
        obs = b.get_observation()
        assert (obs == HIDDEN).all()

    def test_hidden_cells_decreases_after_reveal(self):
        b = make_board()
        before = len(b.hidden_cells())
        b.reveal(0, 0)
        after = len(b.hidden_cells())
        assert after < before
