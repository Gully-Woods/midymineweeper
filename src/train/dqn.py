"""
Deep Q-Network training loop.

Key ideas implemented:
  - Experience replay buffer (store transitions, sample mini-batches)
  - Target network (separate slowly-updated copy of the Q-network)
  - Epsilon-greedy exploration with decay
  - Action masking (only consider hidden cells)

Usage:
    env   = MinesweeperEnv(preset="beginner", dataset="data/beginner_train.npz")
    agent = DQNAgent(env.state_size, env.action_size)
    train(agent, env, episodes=100_000)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from ..model.dqn_model import DQNModel, CNNDQNModel


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
    def __init__(
        self,
        state_size:    int,
        action_size:   int,
        lr:            float = 1e-3,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size:    int   = 64,
        buffer_size:   int   = 10_000,
        target_update: int   = 100,   # steps between target network sync
        model_cls             = None,  # DQNModel or CNNDQNModel (default: CNNDQNModel)
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

        if model_cls is None:
            model_cls = CNNDQNModel
        self.policy_net = model_cls(state_size, action_size)
        self.target_net = model_cls(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()

    def choose_action(self, obs: np.ndarray, valid_actions: list) -> int:
        if random.random() < self.epsilon or not valid_actions:
            return random.choice(valid_actions)
        with torch.no_grad():
            q = self.policy_net(
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            )[0]
        mask = torch.full((self.action_size,), -1e9)
        mask[valid_actions] = q[torch.tensor(valid_actions)]
        return int(mask.argmax())

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        obs_b, act_b, rew_b, next_b, done_b = zip(
            *self.buffer.sample(self.batch_size)
        )

        obs_t  = torch.tensor(np.array(obs_b),  dtype=torch.float32)
        act_t  = torch.tensor(act_b,             dtype=torch.long)
        rew_t  = torch.tensor(rew_b,             dtype=torch.float32)
        next_t = torch.tensor(np.array(next_b),  dtype=torch.float32)
        done_t = torch.tensor(done_b,             dtype=torch.float32)

        q_vals = self.policy_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q  = self.target_net(next_t).max(1)[0]
            targets = rew_t + self.gamma * next_q * (1 - done_t)

        loss = self.loss_fn(q_vals, targets)
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=float("inf")
        ).item()
        self.optimiser.step()

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item(), grad_norm


def train(agent: DQNAgent, env, episodes: int = 100_000, log_every: int = 1000):
    """Basic DQN training loop."""
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
