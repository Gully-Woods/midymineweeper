# MidyMineweeper

A Minesweeper implementation built for reinforcement learning experimentation.

## Goals

1. **Playable game** — human-playable via pygame
2. **RL environment** — clean `reset()` / `step()` interface for Q-learning and DQN agents
3. **Scalable** — beginner (9×9) through expert (30×16) board presets

---

## Project Structure

```
midymineweeper/
├── main.py                        # Entry point — launch pygame game
├── requirements.txt
├── data/
│   └── boards/                    # Saved board configs and Q-tables
├── tests/
│   ├── test_board.py              # Board logic unit tests
│   └── test_env.py                # RL environment unit tests
└── src/
    ├── config/
    │   └── settings.py            # Board presets, display constants
    ├── game/
    │   ├── board.py               # Core logic (mine placement, reveal, flood-fill)
    │   ├── game.py                # State machine (PLAYING / WON / LOST)
    │   └── renderer.py            # pygame display — only file that imports pygame
    ├── env/
    │   └── minesweeper_env.py     # Custom RL environment (no gymnasium required)
    ├── model/
    │   └── dqn_model.py           # DQN neural network (placeholder, requires torch)
    └── train/
        ├── q_learning.py          # Tabular Q-learning agent + training loop
        └── dqn.py                 # DQN agent + training loop (skeleton)
```

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py                      # beginner board
python main.py --preset intermediate
python main.py --preset expert
python main.py --seed 42            # fixed seed for reproducible board
```

**Controls:**
- Left click — reveal cell
- Right click — place / remove flag
- `R` — restart
- Click header — restart

---

## Board Presets

| Preset       | Grid   | Mines |
|--------------|--------|-------|
| beginner     | 9×9    | 10    |
| intermediate | 16×16  | 40    |
| expert       | 16×30  | 99    |

---

## RL Environment

```python
from src.env.minesweeper_env import MinesweeperEnv

env = MinesweeperEnv(preset="beginner", seed=0)
obs = env.reset()          # shape (81,), values -1..9

valid = env.get_valid_actions()        # list of hidden cell indices
obs, reward, done, info = env.step(valid[0])
```

**Observation:** flat `int8` array, one value per cell:
- `-1` hidden
- `0–8` revealed (adjacent mine count)
- `9` flagged

**Action:** flat index `0…rows*cols-1` → reveal that cell

**Rewards:**
| Event | Reward |
|-------|--------|
| Safe reveal | `+1` per newly revealed cell |
| Mine hit | `-10` |
| Win | `+50` bonus |
| Already revealed | `0` |

---

## Running Tests

```bash
python -m pytest tests/
```

---

## Reinforcement Learning Roadmap

- [x] Game environment with gym-style interface
- [ ] Tabular Q-learning (`src/train/q_learning.py`) — good for small boards
- [ ] Deep Q-Network (`src/train/dqn.py` + `src/model/dqn_model.py`) — scales to larger boards
- [ ] Training visualisation (win rate curves, heatmaps of agent action distribution)
- [ ] Curriculum learning (start easy, increase difficulty as agent improves)
