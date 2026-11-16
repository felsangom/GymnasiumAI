"""
Replay buffers do projeto. Todos compartilham a mesma interface:

    store(experience, episode_start=False)
    sample(batch_size, beta) -> (states, actions, rewards, next_states, dones, weights, indices)
    update_priorities(indices, td_errors)
"""
import numpy as np


class _ArrayStorage:
    """Armazenamento circular em arrays NumPy pré-alocados (alocados no 1º store)."""
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.pos = 0
        self.size = 0
        self._data = None

    def _allocate(self, state):
        state = np.asarray(state)
        shape = (self.maxlen,) + state.shape
        self._data = {
            "states": np.zeros(shape, dtype=state.dtype),
            "next_states": np.zeros(shape, dtype=state.dtype),
            "actions": np.zeros((self.maxlen,), dtype=np.int32),
            "rewards": np.zeros((self.maxlen,), dtype=np.float32),
            "dones": np.zeros((self.maxlen,), dtype=np.float32),
        }

    def write(self, experience):
        """Grava a transição e devolve o índice usado."""
        state, action, reward, next_state, done = experience
        if self._data is None:
            self._allocate(state)
        i = self.pos
        self._data["states"][i] = state
        self._data["next_states"][i] = next_state
        self._data["actions"][i] = action
        self._data["rewards"][i] = reward
        self._data["dones"][i] = float(done)
        self.pos = (self.pos + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)
        return i

    def gather(self, indices):
        d = self._data
        return (
            d["states"][indices],
            d["actions"][indices],
            d["rewards"][indices].reshape(-1, 1),
            d["next_states"][indices],
            d["dones"][indices].reshape(-1, 1),
        )

    def __len__(self):
        return self.size


class ReplayBuffer(_ArrayStorage):
    """Replay buffer com amostragem uniforme."""
    def store(self, experience, episode_start=False):
        self.write(experience)

    def sample(self, batch_size, beta=None):
        indices = np.random.randint(0, self.size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return (*self.gather(indices), weights, indices)

    def update_priorities(self, batch_indices, batch_priorities):
        pass


class PrioritizedReplayBuffer(_ArrayStorage):
    """
    Prioritized Experience Replay com Sum Tree: store/sample/update em O(log n).

    A árvore guarda prioridade**alpha. Experiências novas entram com `max_priority`,
    que decai lentamente (`max_priority_decay`) para que um erro TD atípico não
    infle para sempre a prioridade de tudo o que chega depois.
    """
    def __init__(self, maxlen, alpha=0.4, max_priority_decay=0.999):
        super().__init__(maxlen)
        self.alpha = alpha
        self.max_priority_decay = max_priority_decay
        self.capacity = max(2, 1 << (maxlen - 1).bit_length())  # potência de 2
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.max_priority = 1.0

    def _set(self, indices, values):
        """Define folhas e propaga as somas até a raiz."""
        leaves = np.asarray(indices) + self.capacity
        self.tree[leaves] = values
        parents = np.unique(leaves // 2)
        while True:
            self.tree[parents] = self.tree[2 * parents] + self.tree[2 * parents + 1]
            if parents[0] == 1:
                break
            parents = np.unique(parents // 2)

    def store(self, experience, episode_start=False):
        i = self.write(experience)
        self._set([i], [self.max_priority ** self.alpha])

    def sample(self, batch_size, beta=0.4):
        """Amostragem estratificada pela Sum Tree + pesos de Importance Sampling."""
        n = self.size
        total = self.tree[1]
        segment = total / batch_size
        values = (np.arange(batch_size) + np.random.uniform(size=batch_size)) * segment

        idx = np.ones(batch_size, dtype=np.int64)
        while idx[0] < self.capacity:
            left = 2 * idx
            go_right = values > self.tree[left]
            values = values - self.tree[left] * go_right
            idx = left + go_right
        indices = np.minimum(idx - self.capacity, n - 1)

        probs = np.maximum(self.tree[indices + self.capacity] / total, 1e-12)
        weights = (n * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return (*self.gather(indices), weights, indices)

    def update_priorities(self, batch_indices, batch_priorities):
        """Atualiza prioridades com o erro TD absoluto."""
        priorities = np.abs(np.asarray(batch_priorities)).flatten() + 1e-6
        self.max_priority = max(
            self.max_priority * self.max_priority_decay, float(priorities.max())
        )
        self._set(batch_indices, priorities ** self.alpha)


class FrameStackReplayBuffer:
    """
    Replay buffer uniforme para observações de frames empilhados (H, W, stack).

    Em vez de guardar os dois empilhamentos completos (2 × stack frames por
    transição), guarda só o frame mais recente do estado e do próximo estado
    (2 frames) e reconstrói os empilhamentos na amostragem. Para stack=4 isso
    é 4× menos memória.

    Requer `episode_start=True` no primeiro passo de cada episódio e assume o
    comportamento do FrameStackSkipWrapper (no reset o 1º frame é repetido).
    """
    def __init__(self, maxlen, stack=4):
        self.maxlen = maxlen
        self.stack = stack
        self.pos = 0
        self.size = 0
        self._frames = None

    def _allocate(self, state):
        shape = (self.maxlen,) + state.shape[:-1]
        self._frames = np.zeros(shape, dtype=state.dtype)
        self._next_frames = np.zeros(shape, dtype=state.dtype)
        self._age = np.zeros((self.maxlen,), dtype=np.int64)  # passos desde o início do episódio
        self._actions = np.zeros((self.maxlen,), dtype=np.int32)
        self._rewards = np.zeros((self.maxlen,), dtype=np.float32)
        self._dones = np.zeros((self.maxlen,), dtype=np.float32)

    def store(self, experience, episode_start=False):
        state, action, reward, next_state, done = experience
        state = np.asarray(state)
        if self._frames is None:
            self._allocate(state)
        i = self.pos
        self._frames[i] = state[..., -1]
        self._next_frames[i] = np.asarray(next_state)[..., -1]
        if episode_start or self.size == 0:
            self._age[i] = 0
        else:
            self._age[i] = min(self._age[(i - 1) % self.maxlen] + 1, self.stack - 1)
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self.pos = (self.pos + 1) % self.maxlen
        self.size = min(self.size + 1, self.maxlen)

    def sample(self, batch_size, beta=None):
        if self.size < self.maxlen:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            # As (stack-1) posições logo após `pos` teriam antecessores já sobrescritos
            valid = self.maxlen - (self.stack - 1)
            indices = (self.pos + self.stack - 1 + np.random.randint(0, valid, size=batch_size)) % self.maxlen

        back = np.minimum(
            np.arange(self.stack - 1, -1, -1)[None, :], self._age[indices][:, None]
        )
        frame_idx = (indices[:, None] - back) % self.maxlen
        states = np.moveaxis(self._frames[frame_idx], 1, -1)
        next_states = np.concatenate(
            [states[..., 1:], self._next_frames[indices][..., None]], axis=-1
        )
        weights = np.ones(batch_size, dtype=np.float32)
        return (
            states,
            self._actions[indices],
            self._rewards[indices].reshape(-1, 1),
            next_states,
            self._dones[indices].reshape(-1, 1),
            weights,
            indices,
        )

    def update_priorities(self, batch_indices, batch_priorities):
        pass

    def __len__(self):
        return self.size
