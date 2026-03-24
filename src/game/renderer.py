"""
Pygame renderer for Minesweeper.

Only this file imports pygame. All game logic lives in board.py / game.py.
"""

import sys
import pygame
from .game import Game, GameState
from ..config.settings import (
    CELL_SIZE, HEADER_HEIGHT, FPS,
    COLOUR_HIDDEN, COLOUR_REVEALED, COLOUR_MINE, COLOUR_FLAG,
    COLOUR_GRID, COLOUR_HEADER, COLOUR_TEXT, NUMBER_COLOURS,
)

_FONT_CACHE = {}


def _font(size: int):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = pygame.font.SysFont("monospace", size, bold=True)
    return _FONT_CACHE[size]


class Renderer:
    def __init__(self, game: Game):
        self.game = game
        self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        w = self.game.cols * CELL_SIZE
        h = self.game.rows * CELL_SIZE + HEADER_HEIGHT
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("MidyMineweeper")
        self.clock = pygame.time.Clock()

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

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if my < HEADER_HEIGHT:
                        # Click anywhere in header to restart
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
        # Mine counter (left)
        mines_left = self.game.board.mines_remaining()
        surf = _font(28).render(f"Mines: {mines_left}", True, COLOUR_TEXT)
        self.screen.blit(surf, (10, 15))

        # State indicator (centre)
        if self.game.state == GameState.WON:
            label = "YOU WIN!"
            colour = (80, 220, 80)
        elif self.game.state == GameState.LOST:
            label = "BOOM!"
            colour = (220, 80, 80)
        else:
            label = "R = restart"
            colour = COLOUR_TEXT
        surf = _font(22).render(label, True, colour)
        w = self.game.cols * CELL_SIZE
        self.screen.blit(surf, (w // 2 - surf.get_width() // 2, 18))

    def _draw_board(self):
        board = self.game.board
        for r in range(board.rows):
            for c in range(board.cols):
                x = c * CELL_SIZE
                y = r * CELL_SIZE + HEADER_HEIGHT
                val = board.view[r, c]
                self._draw_cell(x, y, val)

    def _draw_cell(self, x: int, y: int, val: int):
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

        if val == -1:  # HIDDEN
            pygame.draw.rect(self.screen, COLOUR_HIDDEN, rect)
        elif val == 9:  # FLAGGED / revealed mine
            if self.game.state == GameState.LOST:
                pygame.draw.rect(self.screen, COLOUR_MINE, rect)
                self._draw_mine_icon(x, y)
            else:
                pygame.draw.rect(self.screen, COLOUR_HIDDEN, rect)
                self._draw_flag_icon(x, y)
        else:  # REVEALED (0-8)
            pygame.draw.rect(self.screen, COLOUR_REVEALED, rect)
            if val > 0:
                colour = NUMBER_COLOURS.get(val, (0, 0, 0))
                surf = _font(22).render(str(val), True, colour)
                self.screen.blit(surf, (
                    x + CELL_SIZE // 2 - surf.get_width() // 2,
                    y + CELL_SIZE // 2 - surf.get_height() // 2,
                ))

        # Grid line
        pygame.draw.rect(self.screen, COLOUR_GRID, rect, 1)

    def _draw_flag_icon(self, x: int, y: int):
        cx = x + CELL_SIZE // 2
        cy = y + CELL_SIZE // 2
        surf = _font(20).render("F", True, COLOUR_FLAG)
        self.screen.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))

    def _draw_mine_icon(self, x: int, y: int):
        cx = x + CELL_SIZE // 2
        cy = y + CELL_SIZE // 2
        surf = _font(20).render("*", True, (0, 0, 0))
        self.screen.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))
