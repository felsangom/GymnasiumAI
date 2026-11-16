import gymnasium as gym
import numpy as np

from common.buffers import PrioritizedReplayBuffer
from common.dqn import DQNAgent, create_q_network
from common.training import TrainConfig, evaluate, run_cli


def _make_env(render_mode=None):
    return gym.make("CartPole-v1", render_mode=render_mode)


def _build_agent(env):
    q, t = create_q_network(4, 2, 16, 8), create_q_network(4, 2, 16, 8)
    t.set_weights(q.get_weights())
    return DQNAgent(q, t, 2, PrioritizedReplayBuffer(2000), 1e-3, 0.99,
                    1.0, 0.05, 500, batch_size=16, warmup_steps=32)


def _config(tmp_path, **kwargs):
    return TrainConfig(name="test", weights_file=tmp_path / "w.weights.h5", log_dir=tmp_path / "logs",
                       num_episodes=6, eval_interval=3, eval_episodes=2, **kwargs)


def test_evaluate_is_deterministic_with_seed():
    env = _make_env()
    agent = _build_agent(env)
    first = evaluate(env, agent, 3, seed=5, verbose=False)
    second = evaluate(env, agent, 3, seed=5, verbose=False)
    assert first == second
    env.close()


def test_train_saves_weights_meta_and_state_then_resumes(tmp_path):
    cfg = _config(tmp_path)
    run_cli(cfg, _make_env, _build_agent, argv=[])
    assert cfg.weights_file.exists() and cfg.meta_file.exists() and cfg.state_file.exists()

    cfg.num_episodes = 9
    run_cli(cfg, _make_env, _build_agent, argv=["--resume"])
    run_cli(cfg, _make_env, _build_agent, argv=["--play", "1"])


def test_existing_better_weights_are_not_overwritten(tmp_path):
    cfg = _config(tmp_path)
    run_cli(cfg, _make_env, _build_agent, argv=[])
    # Finge que os pesos em disco são muito melhores do que qualquer treino curto
    cfg.meta_file.write_text('{"eval_mean": 1e9}')
    before = cfg.weights_file.read_bytes()
    run_cli(cfg, _make_env, _build_agent, argv=["--seed", "7"])
    assert cfg.weights_file.read_bytes() == before
    assert '1e9' in cfg.meta_file.read_text() or '1000000000' in cfg.meta_file.read_text()
