import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from GymCarRacing.safedriving_wrapper import FrameStackSkipWrapper, SafeDrivingWrapper


class _FakeImageEnv(gym.Env):
    """Ambiente de imagem mínimo: cada frame tem o valor do contador de passos."""
    observation_space = spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
    action_space = spaces.Discrete(2)

    def __init__(self, length=10):
        self.length = length
        self.t = 0

    def _obs(self):
        return np.full((8, 8, 3), self.t, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 1.0, self.t >= self.length, False, {"raw_reward": 0.5}


def test_frame_stack_skip_stacks_and_sums_rewards():
    env = FrameStackSkipWrapper(_FakeImageEnv(length=10), frame_skip=4, stack=4)
    obs, _ = env.reset()
    assert obs.shape == (8, 8, 4) and obs.dtype == np.uint8
    assert np.all(obs == 0)

    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 4.0 and info["raw_reward"] == 2.0 and not terminated
    assert [int(obs[0, 0, k]) for k in range(4)] == [0, 0, 0, 4]

    env.step(0)
    obs, reward, terminated, _, info = env.step(0)  # o episódio termina no meio do skip
    assert terminated and reward == 2.0 and info["raw_reward"] == 1.0
    assert [int(obs[0, 0, k]) for k in range(4)] == [0, 4, 8, 10]


def test_safe_driving_wrapper_on_real_env():
    pytest.importorskip("Box2D")
    pytest.importorskip("pygame")
    env = SafeDrivingWrapper(gym.make("CarRacing-v3", continuous=False))
    _, info = env.reset(seed=0)
    assert info["on_track"]

    # Acelerando em linha reta o carro acaba saindo da pista: penalidade + término
    total_raw, steps, terminated, truncated = 0.0, 0, False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(3)
        total_raw += info["raw_reward"]
        steps += 1
    assert terminated and not info["on_track"]
    assert reward < info["raw_reward"] - env.off_track_penalty + 1e-6
    assert steps < 1000
    env.close()
