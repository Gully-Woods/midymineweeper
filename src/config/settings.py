"""
Board presets for Minesweeper.
Each preset defines (rows, cols, num_mines).
"""

PRESETS = {
    "beginner":     {"rows": 9,  "cols": 9,  "num_mines": 10},
    "intermediate": {"rows": 16, "cols": 16, "num_mines": 40},
    "expert":       {"rows": 16, "cols": 30, "num_mines": 99},
}

DEFAULT_PRESET = "beginner"
PRESET_ORDER = ["beginner", "intermediate", "expert"]

# Pygame renderer settings
CELL_SIZE = 40          # pixels per cell
HEADER_HEIGHT = 60      # pixels for top bar (mine count, timer, restart)
FPS = 60

# Colours (R, G, B)
COLOUR_HIDDEN   = (189, 189, 189)
COLOUR_REVEALED = (220, 220, 220)
COLOUR_MINE     = (200,  50,  50)
COLOUR_FLAG     = (230, 120,  30)
COLOUR_WRONG_FLAG = (200,  50,  50)
COLOUR_GRID     = (150, 150, 150)
COLOUR_HEADER   = (100, 100, 100)
COLOUR_TEXT     = (255, 255, 255)

# Classic MS number colours
NUMBER_COLOURS = {
    1: (  0,   0, 220),
    2: (  0, 140,   0),
    3: (200,   0,   0),
    4: (  0,   0, 140),
    5: (140,   0,   0),
    6: (  0, 140, 140),
    7: (  0,   0,   0),
    8: (100, 100, 100),
}
