"""
Pygame renderer for Minesweeper.

Only this file imports pygame. All game logic lives in board.py / game.py.
"""

import sys
import pygame
from .game import Game, GameState
from .board import MINE_EXPOSED, FLAGGED
from ..config.settings import (
    CELL_SIZE, HEADER_HEIGHT, FPS, PRESETS, PRESET_ORDER,
    COLOUR_HIDDEN, COLOUR_REVEALED, COLOUR_MINE, COLOUR_FLAG, COLOUR_WRONG_FLAG,
    COLOUR_GRID, COLOUR_HEADER, COLOUR_TEXT, NUMBER_COLOURS,
)

_FONT_CACHE = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont("monospace", size, bold=True)
    return _FONT_CACHE[size]


class Renderer:
    def __init__(self, game: Game, preset: str = "beginner"):
        self._preset_idx = PRESET_ORDER.index(preset) if preset in PRESET_ORDER else 0
        self.game = game
        self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self._window_size())
        pygame.display.set_caption("MidyMineweeper")
        self.clock = pygame.time.Clock()

    def _window_size(self):
        return (self.game.cols * CELL_SIZE, self.game.rows * CELL_SIZE + HEADER_HEIGHT)

    def _current_preset(self) -> str:
        return PRESET_ORDER[self._preset_idx]

    def _cycle_preset(self):
        self._preset_idx = (self._preset_idx + 1) % len(PRESET_ORDER)
        name = self._current_preset()
        cfg = PRESETS[name]
        self.game = Game(cfg["rows"], cfg["cols"], cfg["num_mines"])
        _FONT_CACHE.clear()
        self.screen = pygame.display.set_mode(self._window_size())

    def run(self):
        """Main event loop — blocks until window is closed."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.game.reset()
                    elif event.key == pygame.K_TAB:
                        self._cycle_preset()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if my < HEADER_HEIGHT:
                        w = self.game.cols * CELL_SIZE
                        if mx > w * 2 // 3:
                            # Right third of header → cycle preset
                            self._cycle_preset()
                        else:
                            # Anywhere else in header → restart same preset
                            self.game.reset()
                    else:
                        col = mx // CELL_SIZE
                        row = (my - HEADER_HEIGHT) // CELL_SIZE
                        if event.button == 1:   # left click → reveal
                            self.game.reveal(row, col)
                        elif event.button == 3: # right click → flag
                            self.game.flag(row, col)

            self._draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        self.screen.fill(COLOUR_HEADER)
        self._draw_header()
        self._draw_board()
        pygame.display.flip()

    def _draw_header(self):
        w = self.game.cols * CELL_SIZE

        # LEFT — mine counter
        mines_surf = _font(22).render(f"Mines: {self.game.board.mines_remaining()}", True, COLOUR_TEXT)
        self.screen.blit(mines_surf, (8, HEADER_HEIGHT // 2 - mines_surf.get_height() // 2))

        # CENTRE — game state
        if self.game.state == GameState.WON:
            label, colour = "YOU WIN!", (80, 220, 80)
        elif self.game.state == GameState.LOST:
            label, colour = "BOOM!", (220, 80, 80)
        else:
            label, colour = "R=restart", COLOUR_TEXT
        state_surf = _font(22).render(label, True, colour)
        self.screen.blit(state_surf, (w // 2 - state_surf.get_width() // 2,
                                      HEADER_HEIGHT // 2 - state_surf.get_height() // 2))

        # RIGHT — preset toggle (TAB or click)
        preset_label = f"{self._current_preset().upper()} [TAB]"
        preset_surf = _font(18).render(preset_label, True, (200, 200, 100))
        self.screen.blit(preset_surf, (w - preset_surf.get_width() - 8,
                                       HEADER_HEIGHT // 2 - preset_surf.get_height() // 2))

    def _draw_board(self):
        board = self.game.board
        for r in range(board.rows):
            for c in range(board.cols):
                x = c * CELL_SIZE
                y = r * CELL_SIZE + HEADER_HEIGHT
                val = board.view[r, c]
                self._draw_cell(x, y, r, c, val)

    def _draw_cell(self, x: int, y: int, r: int, c: int, val: int):
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        lost = self.game.state == GameState.LOST

        if val == MINE_EXPOSED:  # unflagged mine revealed on death
            pygame.draw.rect(self.screen, COLOUR_MINE, rect)
            self._draw_text(x, y, "*", (0, 0, 0))

        elif val == FLAGGED:
            if lost:
                if self.game.board.is_mine(r, c):
                    # Correct flag — keep as flag colour
                    pygame.draw.rect(self.screen, COLOUR_HIDDEN, rect)
                    self._draw_text(x, y, "F", COLOUR_FLAG)
                else:
                    # Wrong flag — grey background, red X
                    pygame.draw.rect(self.screen, COLOUR_HIDDEN, rect)
                    self._draw_text(x, y, "X", COLOUR_WRONG_FLAG)
            else:
                pygame.draw.rect(self.screen, COLOUR_HIDDEN, rect)
                self._draw_text(x, y, "F", COLOUR_FLAG)

        elif val == -1:  # HIDDEN
            pygame.draw.rect(self.screen, COLOUR_HIDDEN, rect)

        else:  # REVEALED 0-8
            pygame.draw.rect(self.screen, COLOUR_REVEALED, rect)
            if val > 0:
                colour = NUMBER_COLOURS.get(val, (0, 0, 0))
                self._draw_text(x, y, str(val), colour, size=22)

        pygame.draw.rect(self.screen, COLOUR_GRID, rect, 1)

    def _draw_text(self, x: int, y: int, text: str, colour, size: int = 20):
        surf = _font(size).render(text, True, colour)
        self.screen.blit(surf, (
            x + CELL_SIZE // 2 - surf.get_width() // 2,
            y + CELL_SIZE // 2 - surf.get_height() // 2,
        ))
