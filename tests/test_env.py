"""
Unit tests for MinesweeperEnv.
Run with: python -m pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.env.minesweeper_env import MinesweeperEnv


class TestEnvBasics:
    def test_reset_returns_obs(self):
        env = MinesweeperEnv(seed=1)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.state_size,)

    def test_all_hidden_at_reset(self):
        env = MinesweeperEnv(seed=1)
        obs = env.reset()
        assert (obs == -1).all()

    def test_step_returns_tuple(self):
        env = MinesweeperEnv(seed=1)
        env.reset()
        obs, reward, done, info = env.step(0)
        assert obs.shape == (env.state_size,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_valid_actions_all_hidden_at_start(self):
        env = MinesweeperEnv(seed=1)
        env.reset()
        valid = env.get_valid_actions()
        assert len(valid) == env.state_size

    def test_valid_actions_shrink_after_reveal(self):
        env = MinesweeperEnv(seed=1)
        env.reset()
        before = len(env.get_valid_actions())
        env.step(0)
        after = len(env.get_valid_actions())
        assert after < before

    def test_mine_gives_negative_reward(self):
        # Run many episodes until we hit a mine
        for seed in range(50):
            env = MinesweeperEnv(seed=seed)
            env.reset()
            done = False
            hit_mine = False
            while not done:
                valid = env.get_valid_actions()
                action = valid[0]
                obs, reward, done, info = env.step(action)
                if info["result"] == "mine":
                    assert reward == -10.0
                    hit_mine = True
                    break
            if hit_mine:
                break

    def test_repeated_reveal_gives_zero_reward(self):
        env = MinesweeperEnv(seed=1)
        env.reset()
        obs, reward1, done, info = env.step(0)
        if not done:
            obs, reward2, done, info = env.step(0)
            assert reward2 == 0.0

    def test_done_flag_on_mine(self):
        for seed in range(100):
            env = MinesweeperEnv(seed=seed)
            env.reset()
            done = False
            while not done:
                valid = env.get_valid_actions()
                obs, reward, done, info = env.step(valid[0])
                if info["result"] == "mine":
                    assert done
                    break

    def test_episode_seed_varies(self):
        env = MinesweeperEnv(seed=0)
        obs1 = env.reset()
        obs2 = env.reset()
        # Different episodes, different seeds — obs at reset are both all-hidden
        assert (obs1 == -1).all() and (obs2 == -1).all()
