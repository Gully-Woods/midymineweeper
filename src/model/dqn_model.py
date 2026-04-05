"""
DQN model definitions.

DQNModel     — flat MLP baseline (81 → 256 → 256 → action_size)
CNNDQNModel  — CNN with one-hot input encoding (recommended)
               One-hot encodes the 9×9 board into 11 channels, runs two
               conv layers to capture spatial patterns, then a linear head
               with dropout for regularisation.
"""

try:
    import torch
    import torch.nn as nn

    class DQNModel(nn.Module):
        """Flat MLP baseline. No spatial awareness — treats board as a vector."""

        def __init__(self, state_size: int, action_size: int, hidden: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_size),
            )

        def forward(self, x):
            return self.net(x.float())


    class CNNDQNModel(nn.Module):
        """
        CNN model with one-hot input encoding and dropout regularisation.

        Input pipeline (inside forward):
          flat obs (batch, 81)  — values: -1 (hidden), 0-8 (revealed), 9 (flagged)
          → one-hot encode      — 11 classes (-1→0, 0→1, …, 9→10)
          → reshape             — (batch, 11, 9, 9)

        Architecture:
          Conv2d(11→32, 3×3, pad=1) → ReLU   keeps spatial size at 9×9
          Conv2d(32→64, 3×3, pad=1) → ReLU   keeps spatial size at 9×9
          Flatten                    → (batch, 5184)
          Linear(5184→256)  → ReLU → Dropout(p)
          Linear(256→action_size)
        """

        # Cell values: -1, 0-8, 9  →  11 classes (shift by +1 for indexing)
        N_CHANNELS = 11
        ROWS       = 9
        COLS       = 9

        def __init__(self, state_size: int, action_size: int,
                     hidden: int = 256, dropout: float = 0.2):
            super().__init__()
            self.action_size = action_size

            self.conv = nn.Sequential(
                nn.Conv2d(self.N_CHANNELS, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            conv_out = 64 * self.ROWS * self.COLS  # 5184

            self.head = nn.Sequential(
                nn.Linear(conv_out, hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden, action_size),
            )

        def forward(self, x):
            # x: (batch, 81) int or float, values in {-1, 0-8, 9}
            x = x.long()
            x = x + 1                                      # shift: -1→0, 0→1, …, 9→10
            x = torch.clamp(x, 0, self.N_CHANNELS - 1)    # safety clamp
            x = torch.zeros(
                x.shape[0], self.N_CHANNELS, self.ROWS * self.COLS,
                device=x.device
            ).scatter_(1, x.unsqueeze(1), 1.0)             # one-hot: (B, 11, 81)
            x = x.view(x.shape[0], self.N_CHANNELS, self.ROWS, self.COLS)
            x = self.conv(x)
            x = x.flatten(1)
            return self.head(x)


except ImportError:
    class DQNModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required. Run: pip install torch")

    class CNNDQNModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required. Run: pip install torch")
