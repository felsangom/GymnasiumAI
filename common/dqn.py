"""
Rede Dueling e agente DQN (Double DQN + target network) compartilhados
entre os ambientes do projeto.
"""
import json
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# =============================================================================
# UTILITÁRIOS
# =============================================================================
def get_render_mode():
    """Retorna "human" somente se a variável de ambiente RENDER=human estiver definida."""
    return "human" if os.environ.get("RENDER", "").lower() == "human" else None


def set_seed(seed):
    """Fixa as sementes de random, NumPy e TensorFlow para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def try_load_weights(models, weights_file, verbose=True):
    """Tenta carregar pesos em todos os modelos. Retorna True em caso de sucesso."""
    weights_file = str(weights_file)
    if not os.path.exists(weights_file):
        if verbose:
            print(f"⚠️ Arquivo de pesos não encontrado: {weights_file}")
        return False
    try:
        for model in models:
            model.load_weights(weights_file)
        if verbose:
            print(f"🚀 Pesos carregados: {weights_file}")
        return True
    except Exception as e:  # formato/arquitetura incompatível, arquivo corrompido etc.
        if verbose:
            print(f"⚠️ Erro ao carregar pesos de {weights_file}: {e}")
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
# AGENTE DQN
# =============================================================================
class DQNAgent:
    """
    Agente DQN com Double DQN, Huber loss ponderada (IS) e target network.

    - Replay buffer injetado (ver common/buffers.py).
    - Alvo de Bellman, passo de gradiente e atualização do target rodam em grafo TF.
    - target_update_interval=None: soft update (Polyak, `tau`) a cada passo.
      target_update_interval=N: cópia completa a cada N passos de treino.
    - beta (Importance Sampling do PER) vai de `beta_start` a 1.0 em `beta_steps`
      passos de treino. Se `beta_steps` for None, usa 0.6 * epsilon_decay_steps.
    """
    def __init__(
        self, q_network, target_network, action_space_size, replay_buffer,
        learning_rate, discount_factor,
        epsilon_start, epsilon_end, epsilon_decay_steps,
        batch_size, tau=0.01, warmup_steps=1000,
        target_update_interval=None, clipnorm=10.0,
        beta_start=0.4, beta_steps=None
    ):
        self.q_network = q_network
        self.target_network = target_network
        self.action_space_size = action_space_size
        self.replay_buffer = replay_buffer
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.beta_start = beta_start
        self.beta_steps = beta_steps if beta_steps else max(1, int(0.6 * epsilon_decay_steps))
        self.train_step = 0
        self.epsilon = epsilon_start
        self.beta = beta_start
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    # --- Agendamentos --------------------------------------------------------
    def set_epsilon_start(self, epsilon_start):
        """Redefine o epsilon inicial (ex.: ao continuar de pesos já treinados)."""
        self.epsilon_start = epsilon_start
        self._refresh_schedules()

    def _refresh_schedules(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * float(
            np.exp(-self.train_step / self.epsilon_decay_steps)
        )
        self.beta = min(
            1.0, self.beta_start + (1.0 - self.beta_start) * self.train_step / self.beta_steps
        )

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

    def store_experience(self, state, action, reward, next_state, terminal, episode_start=False):
        """`terminal` deve ser apenas `terminated` (truncamento NÃO zera o bootstrap)."""
        self.replay_buffer.store(
            (state, action, reward, next_state, terminal), episode_start=episode_start
        )

    # --- Treinamento ---------------------------------------------------------
    def update_q_network(self):
        """Executa um passo de treino. Retorna (loss, avg_q) ou None durante o warmup."""
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size, self.beta)

        loss, td_errors, avg_q = self._train_step(
            states, actions, rewards, next_states, dones, weights
        )
        self.replay_buffer.update_priorities(indices, td_errors.numpy())

        self.train_step += 1
        self._refresh_schedules()

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

    # --- Estado de treino (para retomar) --------------------------------------
    def save_state(self, path, **extra):
        """
        Salva redes online/target, estado do otimizador, contador de passos e
        `extra` (metadados em JSON). O replay buffer NÃO é salvo: ao retomar,
        ele é preenchido de novo durante o warmup.
        """
        self.optimizer.build(self.q_network.trainable_variables)
        arrays = {}
        for prefix, values in (
            ("q", self.q_network.get_weights()),
            ("t", self.target_network.get_weights()),
            ("o", [v.numpy() for v in self.optimizer.variables]),
        ):
            arrays[f"{prefix}_count"] = len(values)
            for i, value in enumerate(values):
                arrays[f"{prefix}_{i}"] = value
        meta = dict(extra, train_step=self.train_step, epsilon_start=self.epsilon_start)
        tmp = str(path) + ".tmp.npz"
        np.savez(tmp, meta=json.dumps(meta), **arrays)
        os.replace(tmp, str(path))

    def load_state(self, path):
        """Restaura o estado salvo por `save_state`. Retorna o dicionário de metadados."""
        with np.load(str(path), allow_pickle=False) as data:
            def values(prefix):
                return [data[f"{prefix}_{i}"] for i in range(int(data[f"{prefix}_count"]))]

            self.q_network.set_weights(values("q"))
            self.target_network.set_weights(values("t"))
            self.optimizer.build(self.q_network.trainable_variables)
            opt_values = values("o")
            opt_vars = self.optimizer.variables
            if len(opt_values) == len(opt_vars) and all(
                tuple(v.shape) == a.shape for v, a in zip(opt_vars, opt_values)
            ):
                for var, value in zip(opt_vars, opt_values):
                    var.assign(value)
            else:
                print("⚠️ Estado do otimizador incompatível; reiniciando o otimizador.")
            meta = json.loads(str(data["meta"]))
        self.train_step = int(meta["train_step"])
        self.epsilon_start = float(meta["epsilon_start"])
        self._refresh_schedules()
        return meta
