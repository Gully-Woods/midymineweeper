"""
Custom Minesweeper RL environment.

No gymnasium dependency — exposes reset() / step() / get_valid_actions()
so any Q-learning or DQN training loop can plug straight in.

Observation:
  Flat numpy int8 array of shape (rows * cols,)
  Values: -1 (hidden), 0-8 (revealed + adjacent mine count), 9 (flagged)

Action:
  Integer in [0, rows*cols). Maps to (row, col) via divmod(action, cols).

Rewards:
  +1   per safe cell newly revealed
  -10  on hitting a mine (episode ends)
  +50  on winning (all safe cells revealed)
   0   for clicking an already-revealed / flagged cell
"""

import numpy as np
from ..game.game import Game, GameState
from ..config.settings import PRESETS, DEFAULT_PRESET


class MinesweeperEnv:
    def __init__(self, preset: str = DEFAULT_PRESET, seed: int = None):
        cfg = PRESETS[preset]
        self.rows       = cfg["rows"]
        self.cols       = cfg["cols"]
        self.num_mines  = cfg["num_mines"]
        self._base_seed = seed
        self._episode   = 0
        self._game: Game = None
        self.reset()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns initial observation."""
        seed = None if self._base_seed is None else self._base_seed + self._episode
        self._game = Game(self.rows, self.cols, self.num_mines, seed=seed)
        self._episode += 1
        self._prev_revealed = 0
        return self._obs()

    def step(self, action: int) -> tuple:
        """
        Apply action (cell index) and return (obs, reward, done, info).

        obs    — numpy int8 array, shape (rows*cols,)
        reward — float
        done   — bool
        info   — dict with extra diagnostics
        """
        row, col = divmod(action, self.cols)
        result = self._game.reveal(row, col)

        obs = self._obs()
        done = self._game.is_over

        if result == "mine":
            reward = -10.0
        elif result == "already_revealed":
            reward = 0.0
        else:
            # Reward proportional to newly revealed cells
            now_revealed = int((self._game.board.view >= 0).sum()) - \
                           int((self._game.board.view == 9).sum())
            newly = now_revealed - self._prev_revealed
            self._prev_revealed = now_revealed
            reward = float(newly)
            if self._game.state == GameState.WON:
                reward += 50.0

        info = {
            "state":        self._game.state.name,
            "result":       result,
            "mines_left":   self._game.board.mines_remaining(),
        }
        return obs, reward, done, info

    def get_valid_actions(self) -> list:
        """Return flat indices of all currently hidden cells."""
        return self._game.board.hidden_cells()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return self._game.board.get_observation()

    @property
    def action_size(self) -> int:
        return self.rows * self.cols

    @property
    def state_size(self) -> int:
        return self.rows * self.cols
