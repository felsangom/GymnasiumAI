from collections import deque

import numpy as np

from common.buffers import FrameStackReplayBuffer, PrioritizedReplayBuffer, ReplayBuffer


def _transition(i, done=False):
    return (np.full(3, i, dtype=np.float32), i % 2, float(i), np.full(3, i + 1, dtype=np.float32), done)


def test_uniform_buffer_is_circular_and_keeps_shapes():
    buffer = ReplayBuffer(5)
    for i in range(8):
        buffer.store(_transition(i, done=(i == 7)))
    assert len(buffer) == 5
    states, actions, rewards, next_states, dones, weights, indices = buffer.sample(16)
    assert states.shape == (16, 3) and next_states.shape == (16, 3)
    assert rewards.shape == (16, 1) and dones.shape == (16, 1)
    assert actions.dtype == np.int32 and np.all(weights == 1.0)
    assert states.min() >= 3  # as transições 0..2 foram sobrescritas
    np.testing.assert_array_equal(next_states, states + 1)


def test_prioritized_sampling_follows_priorities():
    np.random.seed(0)
    buffer = PrioritizedReplayBuffer(10, alpha=1.0)
    for i in range(10):
        buffer.store(_transition(i))
    priorities = np.arange(1, 11, dtype=np.float64)
    buffer.update_priorities(np.arange(10), priorities)

    counts = np.zeros(10)
    for _ in range(2000):
        counts += np.bincount(buffer.sample(8)[-1], minlength=10)
    np.testing.assert_allclose(counts / counts.sum(), priorities / priorities.sum(), atol=0.01)


def test_prioritized_tree_stays_consistent_after_wraparound():
    np.random.seed(0)
    buffer = PrioritizedReplayBuffer(6, alpha=0.6)  # capacidade interna arredonda para 8
    for i in range(20):
        buffer.store(_transition(i))
        if len(buffer) >= 4:
            *_, weights, indices = buffer.sample(4, beta=0.5)
            assert indices.max() < len(buffer)
            assert np.all(weights > 0) and weights.max() == 1.0
            buffer.update_priorities(indices, np.random.rand(4, 1))
    leaves = buffer.tree[buffer.capacity:]
    assert np.isclose(buffer.tree[1], leaves.sum())
    assert np.all(leaves[len(buffer):] == 0)


def test_max_priority_decays_after_outlier():
    buffer = PrioritizedReplayBuffer(4, max_priority_decay=0.5)
    for i in range(4):
        buffer.store(_transition(i))
    buffer.update_priorities([0], [100.0])
    assert buffer.max_priority >= 100.0
    for _ in range(10):
        buffer.update_priorities([1], [0.1])
    assert buffer.max_priority < 1.0


def test_frame_stack_buffer_reconstructs_stacks():
    """Compara os empilhamentos reconstruídos com os produzidos passo a passo."""
    np.random.seed(0)
    stack, maxlen = 4, 37
    buffer = FrameStackReplayBuffer(maxlen, stack)
    reference = {}  # índice no buffer -> (state, next_state, action)
    frame_id = 0

    for episode in range(12):
        frames = deque(maxlen=stack)
        first = np.full((5, 5), frame_id, dtype=np.uint8)
        frame_id += 1
        for _ in range(stack):
            frames.append(first)
        state = np.stack(frames, axis=-1)
        for step in range(np.random.randint(1, 9)):
            frames.append(np.full((5, 5), frame_id % 256, dtype=np.uint8))
            frame_id += 1
            next_state = np.stack(frames, axis=-1)
            reference[buffer.pos] = (state, next_state, frame_id)
            buffer.store((state, frame_id, 1.0, next_state, False), episode_start=(step == 0))
            state = next_state

            if len(buffer) >= 8:
                states, actions, _, next_states, _, _, indices = buffer.sample(8)
                for k, idx in enumerate(indices):
                    ref_state, ref_next, ref_action = reference[idx]
                    assert actions[k] == ref_action
                    np.testing.assert_array_equal(states[k], ref_state)
                    np.testing.assert_array_equal(next_states[k], ref_next)

    assert len(buffer) == maxlen  # garante que o caso circular foi exercitado
