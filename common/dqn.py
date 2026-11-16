"""
Componentes compartilhados entre os agentes DQN do projeto.

Contém: camada Dueling, Replay Buffers (priorizado com Sum Tree e uniforme),
o agente DQN (Double DQN + soft/hard target update) e o loop de treino/avaliação.
"""
import datetime
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# =============================================================================
# UTILITÁRIOS
# =============================================================================
def get_render_mode():
    """
    Retorna "human" somente se a variável de ambiente RENDER=human estiver definida.
    Por padrão não abre janela (treino mais rápido e utilizável em servidores).
    """
    return "human" if os.environ.get("RENDER", "").lower() == "human" else None


def set_seed(seed):
    """Fixa as sementes de random, NumPy e TensorFlow para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def try_load_weights(models, weights_file):
    """Tenta carregar pesos em todos os modelos. Retorna True em caso de sucesso."""
    if not os.path.exists(weights_file):
        print(f"⚠️ Arquivo de pesos não encontrado: {weights_file}. Iniciando do zero.")
        return False
    try:
        for model in models:
            model.load_weights(weights_file)
        print(f"🚀 Pesos carregados: {weights_file}")
        return True
    except Exception as e:  # formato/arquitetura incompatível, arquivo corrompido etc.
        print(f"⚠️ Erro ao carregar pesos: {e}. Iniciando do zero.")
        return False


# =============================================================================
# REDE NEURAL (DUELING)
# =============================================================================
class DuelingAggregation(tf.keras.layers.Layer):
    """
    Agregação Dueling: Q(s,a) = V(s) + (A(s,a) - média(A(s,a))).
    A subtração da média garante a identificabilidade do modelo.
    """
    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))


def create_q_network(state_space_size, action_space_size, shared_units=128, branch_units=64):
    """Dueling Q-Network (API funcional do Keras) para estados vetoriais."""
    inputs = Input(shape=(state_space_size,))
    shared = Dense(shared_units, activation='relu')(inputs)
    shared = Dense(shared_units, activation='relu')(shared)

    value = Dense(branch_units, activation='relu')(shared)
    value = Dense(1, activation='linear')(value)

    advantage = Dense(branch_units, activation='relu')(shared)
    advantage = Dense(action_space_size, activation='linear')(advantage)

    outputs = DuelingAggregation()([value, advantage])
    return Model(inputs=inputs, outputs=outputs)


# =============================================================================
# REPLAY BUFFERS
# =============================================================================
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


class ReplayBuffer(_ArrayStorage):
    """Replay Buffer com amostragem uniforme (mesma interface do PER)."""
    def store(self, experience):
        self.write(experience)

    def sample(self, batch_size, beta=None):
        indices = np.random.randint(0, self.size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return (*self.gather(indices), weights, indices)

    def update_priorities(self, batch_indices, batch_priorities):
        pass

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(_ArrayStorage):
    """
    Prioritized Experience Replay com Sum Tree: store/sample/update em O(log n).
    A árvore guarda prioridade**alpha; `max_priority` é mantido incrementalmente.
    """
    def __init__(self, maxlen, alpha=0.4):
        super().__init__(maxlen)
        self.alpha = alpha
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

    def store(self, experience):
        """Novas experiências recebem a prioridade máxima conhecida."""
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
        self.max_priority = max(self.max_priority, float(priorities.max()))
        self._set(batch_indices, priorities ** self.alpha)

    def __len__(self):
        return self.size


# =============================================================================
# AGENTE DQN
# =============================================================================
class DQNAgent:
    """
    Agente DQN com Double DQN, Huber loss ponderada (IS) e target network.

    - Replay buffer injetado (priorizado ou uniforme).
    - Alvo de Bellman, passo de gradiente e atualização do target rodam
      dentro de um único tf.function (sem idas e voltas para NumPy).
    - target_update_interval=None: soft update (Polyak, `tau`) a cada passo.
      target_update_interval=N: cópia completa a cada N passos de treino.
    """
    def __init__(
        self, q_network, target_network, action_space_size, replay_buffer,
        learning_rate, discount_factor,
        epsilon_start, epsilon_end, epsilon_decay_steps,
        batch_size, tau=0.01, warmup_steps=1000,
        target_update_interval=None, clipnorm=10.0
    ):
        self.q_network = q_network
        self.target_network = target_network
        self.action_space_size = action_space_size
        self.replay_buffer = replay_buffer
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.train_step = 0
        self.beta = 0.4  # annealing do Importance Sampling (só afeta o PER)
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    # --- Interação com o ambiente -------------------------------------------
    def choose_action(self, state, env):
        """Política epsilon-greedy."""
        if random.random() < self.epsilon:
            return env.action_space.sample()
        return self.greedy_action(state)

    def greedy_action(self, state):
        """Ação de maior valor Q (sem exploração)."""
        q_values = self._predict_q_values(np.expand_dims(state, axis=0))[0]
        return int(np.argmax(q_values))

    def store_experience(self, state, action, reward, next_state, terminal):
        """`terminal` deve ser apenas `terminated` (truncamento NÃO zera o bootstrap)."""
        self.replay_buffer.store((state, action, reward, next_state, terminal))

    # --- Treinamento ---------------------------------------------------------
    def update_q_network(self):
        """Executa um passo de treino. Retorna (loss, avg_q) ou None durante o warmup."""
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        self.beta = min(1.0, 0.4 + self.train_step / self.epsilon_decay_steps)
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size, self.beta)

        loss, td_errors, avg_q = self._train_step(
            states, actions, rewards, next_states, dones, weights
        )
        self.replay_buffer.update_priorities(indices, td_errors.numpy())

        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.train_step / self.epsilon_decay_steps
        )
        self.train_step += 1

        if self.target_update_interval is None:
            self._update_target(tf.constant(self.tau, dtype=tf.float32))
        elif self.train_step % self.target_update_interval == 0:
            self._update_target(tf.constant(1.0, dtype=tf.float32))

        return float(loss), float(avg_q)

    @tf.function
    def _predict_q_values(self, state):
        return self.q_network(state, training=False)

    @tf.function
    def _update_target(self, tau):
        """theta_target = tau * theta_online + (1 - tau) * theta_target."""
        for t_var, q_var in zip(self.target_network.variables, self.q_network.variables):
            t_var.assign(tau * q_var + (1.0 - tau) * t_var)

    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, weights):
        """Alvo Double DQN + Huber loss ponderada + gradiente, tudo em um grafo."""
        # Rede online escolhe a ação; rede target avalia o valor
        next_actions = tf.argmax(self.q_network(next_states, training=False), axis=1)
        next_q_target = self.target_network(next_states, training=False)
        next_mask = tf.one_hot(next_actions, depth=self.action_space_size)
        next_q = tf.reduce_sum(next_q_target * next_mask, axis=1, keepdims=True)
        targets = tf.stop_gradient(rewards + self.discount_factor * next_q * (1.0 - dones))

        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, depth=self.action_space_size)
            predicted = tf.reduce_sum(action_masks * q_values, axis=1, keepdims=True)

            td_errors = tf.abs(targets - predicted)
            huber = tf.keras.losses.Huber(reduction='none')(targets, predicted)
            loss = tf.reduce_mean(huber * tf.cast(weights, tf.float32))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        return loss, td_errors, tf.reduce_mean(q_values)


# =============================================================================
# LOOP DE TREINO E AVALIAÇÃO
# =============================================================================
def train(
    env, agent, *, num_episodes, log_dir, weights_file=None,
    reward_window_size=100, target_mean=None, consecutive_required=1,
    reward_clip=None, seed=None
):
    """
    Treina o agente.

    - Armazena apenas `terminated` como terminal (truncamento mantém o bootstrap).
    - Checkpoint: salva os pesos quando a média móvel (janela cheia) melhora.
    - Parada: média móvel >= target_mean por `consecutive_required` episódios seguidos.
    - reward_clip: (min, max) aplicado somente à recompensa armazenada para treino.
    """
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(os.path.join(str(log_dir), current_time))

    if seed is not None:
        env.action_space.seed(seed)

    reward_window = deque(maxlen=reward_window_size)
    best_mean_reward = -np.inf
    consecutive_successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed if (seed is not None and episode == 0) else None)
        total_reward = 0.0
        done = False
        losses, qs = [], []

        while not done:
            action = agent.choose_action(state, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            train_reward = float(np.clip(reward, *reward_clip)) if reward_clip else reward
            agent.store_experience(state, action, train_reward, next_state, terminated)

            result = agent.update_q_network()
            if result:
                losses.append(result[0])
                qs.append(result[1])

            state = next_state
            total_reward += reward

        reward_window.append(total_reward)
        mean_reward = float(np.mean(reward_window))
        window_full = len(reward_window) == reward_window_size

        with summary_writer.as_default():
            tf.summary.scalar('Reward', total_reward, step=episode)
            tf.summary.scalar('Mean_Reward', mean_reward, step=episode)
            tf.summary.scalar('Epsilon', agent.epsilon, step=episode)
            if losses:
                tf.summary.scalar('Loss', np.mean(losses), step=episode)
                tf.summary.scalar('Avg_Q_Value', np.mean(qs), step=episode)

        if weights_file and window_full and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            agent.q_network.save_weights(weights_file)
            print(f"🏆 Novo melhor modelo: média={mean_reward:.1f} | Pesos salvos.")

        print(
            f"Ep {episode + 1:5d} | Reward: {total_reward:8.1f} | "
            f"Média({reward_window_size}): {mean_reward:7.1f} | "
            f"ε: {agent.epsilon:.3f} | Buffer: {len(agent.replay_buffer):6d}"
        )

        if target_mean is not None and window_full and mean_reward >= target_mean:
            consecutive_successes += 1
            if consecutive_successes >= consecutive_required:
                print(f"\n✅ Problema resolvido em {episode + 1} episódios! "
                      f"Média final: {mean_reward:.1f}")
                break
        else:
            consecutive_successes = 0


def evaluate(env, agent, episodes=10, seed=None):
    """Avalia a política puramente gulosa (epsilon=0). Retorna a recompensa média."""
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep if seed is not None else None)
        done, total = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.greedy_action(state))
            done = terminated or truncated
            total += reward
        rewards.append(total)
    mean = float(np.mean(rewards))
    print(f"📊 Avaliação greedy ({episodes} episódios): média={mean:.1f} | "
          f"min={min(rewards):.1f} | max={max(rewards):.1f}")
    return mean
