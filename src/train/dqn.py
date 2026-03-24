"""
Deep Q-Network training loop (placeholder / skeleton).

DQN replaces the Q-table with a neural network, making it feasible to
learn on larger boards where the state space is too big for tabular Q.

Key ideas (to implement):
  - Experience replay buffer (store transitions, sample mini-batches)
  - Target network (separate slowly-updated copy of the Q-network)
  - Epsilon-greedy exploration with decay
  - Action masking (only consider hidden cells)

This file is intentionally a skeleton — ready to fill in once the
Q-learning baseline is working.

Usage (future):
    env   = MinesweeperEnv(preset="intermediate")
    agent = DQNAgent(env.state_size, env.action_size)
    train(agent, env, episodes=100_000)
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Skeleton DQN agent. Fill in train_step() once torch is integrated.
    """
    def __init__(
        self,
        state_size:   int,
        action_size:  int,
        lr:           float = 1e-3,
        gamma:        float = 0.95,
        epsilon:      float = 1.0,
        epsilon_min:  float = 0.05,
        epsilon_decay:float = 0.995,
        batch_size:   int   = 64,
        buffer_size:  int   = 10_000,
        target_update:int   = 100,   # steps between target network sync
    ):
        self.action_size   = action_size
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.steps         = 0

        self.buffer = ReplayBuffer(buffer_size)

        # TODO: initialise policy_net and target_net (DQNModel) and optimiser
        # from src.model.dqn_model import DQNModel
        # self.policy_net = DQNModel(state_size, action_size)
        # self.target_net = DQNModel(state_size, action_size)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.optimiser = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def choose_action(self, obs: np.ndarray, valid_actions: list) -> int:
        if np.random.rand() < self.epsilon or not valid_actions:
            return np.random.choice(valid_actions)
        # TODO: forward pass through policy_net, mask invalid, argmax
        return np.random.choice(valid_actions)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        # TODO: sample batch, compute targets, backprop
        self.steps += 1
        if self.steps % self.target_update == 0:
            pass  # TODO: target_net.load_state_dict(policy_net.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(agent: DQNAgent, env, episodes: int = 100_000, log_every: int = 1000):
    """Basic DQN training loop skeleton."""
    wins = 0
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(obs, valid)
            next_obs, reward, done, info = env.step(action)
            agent.push(obs, action, reward, next_obs, done)
            agent.train_step()
            obs = next_obs
        if info["state"] == "WON":
            wins += 1
        if ep % log_every == 0:
            print(f"Episode {ep}/{episodes}  wins={wins}  ε={agent.epsilon:.3f}")
            wins = 0
