"""
Thin state machine wrapping Board.

States: PLAYING → WON | LOST
"""

from enum import Enum, auto
from .board import Board


class GameState(Enum):
    PLAYING = auto()
    WON     = auto()
    LOST    = auto()


class Game:
    def __init__(self, rows: int, cols: int, num_mines: int, seed: int = None):
        self.rows = rows
        self.cols = cols
        self.num_mines = num_mines
        self.seed = seed
        self.board = Board(rows, cols, num_mines, seed)
        self.state = GameState.PLAYING

    def reveal(self, row: int, col: int) -> str:
        if self.state != GameState.PLAYING:
            return "game_over"

        result = self.board.reveal(row, col)

        if result == "mine":
            self.board.reveal_all_mines()
            self.state = GameState.LOST
        elif self.board.is_won():
            self.state = GameState.WON

        return result

    def flag(self, row: int, col: int):
        if self.state == GameState.PLAYING:
            self.board.flag(row, col)

    def reset(self, seed: int = None):
        self.seed = seed
        self.board = Board(self.rows, self.cols, self.num_mines, seed)
        self.state = GameState.PLAYING

    @property
    def is_over(self) -> bool:
        return self.state != GameState.PLAYING
