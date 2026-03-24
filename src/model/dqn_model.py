"""
Deep Q-Network model (placeholder).

A simple fully-connected network that maps a board observation
(flat int array of length state_size) to Q-values for each action.

Usage (future):
    model = DQNModel(state_size=81, action_size=81)
    q_values = model(torch.tensor(obs, dtype=torch.float32))
"""

try:
    import torch
    import torch.nn as nn

    class DQNModel(nn.Module):
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

except ImportError:
    # torch not installed yet — placeholder so imports don't fail
    class DQNModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for DQNModel. Run: pip install torch")
