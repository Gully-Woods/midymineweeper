"""
Tabular Q-learning agent for Minesweeper (placeholder / skeleton).

Tabular Q-learning works by maintaining a Q-table: a mapping from
(state, action) → expected future reward. For a 9×9 board the raw state
space is enormous (10^81 possible observations), so in practice this
agent works best on tiny boards or with a simplified state representation.

This file is intentionally a skeleton — ready to fill in once the game
env is working and you are ready to train.

Usage (future):
    env = MinesweeperEnv(preset="beginner")
    agent = QLearningAgent(env.state_size, env.action_size)
    train(agent, env, episodes=10_000)
"""

import numpy as np
import json
from pathlib import Path


class QLearningAgent:
    def __init__(
        self,
        state_size:  int,
        action_size: int,
        alpha:   float = 0.1,    # learning rate
        gamma:   float = 0.95,   # discount factor
        epsilon: float = 1.0,    # exploration rate (decays over time)
        epsilon_min:   float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.action_size   = action_size
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: keyed by tuple(obs) → np.array of Q-values per action
        # NOTE: for large boards, replace with a dict or neural net (DQN)
        self.q_table: dict = {}

    def _key(self, obs: np.ndarray) -> tuple:
        return tuple(obs.tolist())

    def _q(self, obs: np.ndarray) -> np.ndarray:
        k = self._key(obs)
        if k not in self.q_table:
            self.q_table[k] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[k]

    def choose_action(self, obs: np.ndarray, valid_actions: list) -> int:
        """Epsilon-greedy action selection over valid (hidden) cells only."""
        if np.random.rand() < self.epsilon or not valid_actions:
            return np.random.choice(valid_actions)
        q = self._q(obs)
        # Mask invalid actions
        masked = np.full(self.action_size, -np.inf)
        masked[valid_actions] = q[valid_actions]
        return int(np.argmax(masked))

    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        """Standard Q-learning Bellman update."""
        q      = self._q(obs)
        q_next = self._q(next_obs)
        target = reward + (0.0 if done else self.gamma * q_next.max())
        q[action] += self.alpha * (target - q[action])

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        serialisable = {str(k): v.tolist() for k, v in self.q_table.items()}
        Path(path).write_text(json.dumps(serialisable))

    def load(self, path: str):
        data = json.loads(Path(path).read_text())
        self.q_table = {
            tuple(map(int, k.strip("()").split(", "))): np.array(v, dtype=np.float32)
            for k, v in data.items()
        }


def train(agent: QLearningAgent, env, episodes: int = 10_000, log_every: int = 500):
    """Basic training loop."""
    wins = 0
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(obs, valid)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
        if info["state"] == "WON":
            wins += 1
        if ep % log_every == 0:
            print(f"Episode {ep}/{episodes}  wins={wins}  ε={agent.epsilon:.3f}")
            wins = 0
