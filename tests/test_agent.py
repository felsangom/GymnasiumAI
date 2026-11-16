import numpy as np

from common.buffers import PrioritizedReplayBuffer
from common.dqn import DQNAgent, create_q_network


def _make_agent(tau=0.5, **kwargs):
    q, t = create_q_network(4, 2, 16, 8), create_q_network(4, 2, 16, 8)
    t.set_weights(q.get_weights())
    return DQNAgent(q, t, 2, PrioritizedReplayBuffer(256), 1e-3, 0.99,
                    1.0, 0.1, 100, batch_size=8, tau=tau, warmup_steps=16, **kwargs)


def _fill(agent, n, terminal=False):
    rng = np.random.default_rng(0)
    for _ in range(n):
        s, s2 = rng.normal(size=4).astype(np.float32), rng.normal(size=4).astype(np.float32)
        agent.store_experience(s, int(rng.integers(2)), 1.0, s2, terminal)


def test_no_training_during_warmup():
    agent = _make_agent()
    _fill(agent, 10)
    assert agent.update_q_network() is None
    assert agent.train_step == 0 and agent.epsilon == 1.0


def test_training_updates_schedules_and_target():
    agent = _make_agent()
    _fill(agent, 32)
    target_before = [w.copy() for w in agent.target_network.get_weights()]
    for _ in range(20):
        loss, avg_q = agent.update_q_network()
        assert np.isfinite(loss) and np.isfinite(avg_q)
    assert agent.train_step == 20
    assert 0.1 < agent.epsilon < 1.0
    assert 0.4 < agent.beta <= 1.0
    assert any(not np.allclose(a, b) for a, b in zip(target_before, agent.target_network.get_weights()))


def test_terminal_transitions_do_not_bootstrap():
    """Com todas as transições terminais e recompensa 1, Q(s,a) deve convergir para 1."""
    agent = _make_agent()
    _fill(agent, 64, terminal=True)
    for _ in range(600):
        agent.update_q_network()
    states = agent.replay_buffer.gather(np.arange(64))[0]
    actions = agent.replay_buffer.gather(np.arange(64))[1]
    q = agent.q_network(states).numpy()[np.arange(64), actions]
    assert np.abs(q - 1.0).mean() < 0.2


def test_hard_target_update_interval():
    agent = _make_agent(target_update_interval=5)
    _fill(agent, 32)
    for _ in range(4):
        agent.update_q_network()
    assert any(not np.allclose(a, b) for a, b in
               zip(agent.q_network.get_weights(), agent.target_network.get_weights()))
    agent.update_q_network()
    for a, b in zip(agent.q_network.get_weights(), agent.target_network.get_weights()):
        np.testing.assert_allclose(a, b)


def test_save_and_load_state_roundtrip(tmp_path):
    agent = _make_agent()
    _fill(agent, 32)
    for _ in range(10):
        agent.update_q_network()
    path = tmp_path / "state.npz"
    agent.save_state(path, episode=7, run_best=12.5)

    restored = _make_agent()
    meta = restored.load_state(path)
    assert meta["episode"] == 7 and meta["run_best"] == 12.5
    assert restored.train_step == agent.train_step
    assert np.isclose(restored.epsilon, agent.epsilon)
    for a, b in zip(agent.q_network.get_weights(), restored.q_network.get_weights()):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(agent.target_network.get_weights(), restored.target_network.get_weights()):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(agent.optimizer.variables, restored.optimizer.variables):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
