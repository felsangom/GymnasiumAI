import datetime
import random

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =============================================================================
# HIPERPARÂMETROS — Ajustados para estabilidade e convergência
# =============================================================================
learning_rate = 0.0005
discount_factor = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_steps = 5000
replay_buffer_size = 50000
batch_size = 64
tau = 0.01
warmup_steps = 1000
weights_file = "cartpole_dqn_weights.weights.h5"
load_existing_weights = False

# Configuração do TensorBoard para monitoramento do treinamento
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/dqn/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

# Inicialização do ambiente Gymnasium
env = gym.make('CartPole-v1', render_mode="human")
state_space_size = env.observation_space.shape[-1]
action_space_size = env.action_space.n

class DuelingAggregation(tf.keras.layers.Layer):
    """
    Camada de agregação para a arquitetura Dueling DQN.
    Combina o valor do estado (V) com a vantagem das ações (A) usando a fórmula:
    Q(s,a) = V(s) + (A(s,a) - média(A(s,a))).
    A subtração da média garante a estabilidade e a identificabilidade do modelo.
    """
    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))


def create_q_network(state_space_size, action_space_size):
    """
    Constrói a arquitetura Dueling Q-Network usando a API Funcional do Keras.
    Define duas ramificações após as camadas compartilhadas: uma para estimar
    o valor do estado e outra para estimar a vantagem de cada ação.
    """
    inputs = Input(shape=(state_space_size,))
    shared = Dense(128, activation='relu')(inputs)
    shared = Dense(128, activation='relu')(shared)

    # Ramo de Valor (State Value)
    value = Dense(64, activation='relu')(shared)
    value = Dense(1, activation='linear')(value)

    # Ramo de Vantagem (Action Advantage)
    advantage = Dense(64, activation='relu')(shared)
    advantage = Dense(action_space_size, activation='linear')(advantage)

    outputs = DuelingAggregation()([value, advantage])
    model = Model(inputs=inputs, outputs=outputs)
    return model


class PrioritizedReplayBuffer:
    """
    Implementação de Prioritized Experience Replay (PER).
    Armazena experiências e permite a amostragem baseada na importância (erro TD).
    Utiliza o hiperparâmetro alpha para controlar o nível de priorização.
    """
    def __init__(self, maxlen, alpha=0.4):
        self.maxlen = maxlen
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((maxlen,), dtype=np.float32)
        self.pos = 0

    def store(self, experience):
        """
        Armazena uma nova experiência no buffer circular.
        Novas experiências recebem inicialmente a prioridade máxima para garantir processamento.
        """
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.maxlen

    def sample(self, batch_size, beta=0.4):
        """
        Realiza a amostragem de um lote de experiências com base em suas prioridades.
        Calcula os pesos de Importance Sampling (IS) para corrigir o viés da amostragem.
        """
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            weights,
            indices
        )

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Atualiza as prioridades no buffer após uma etapa de treinamento.
        A prioridade é baseada no erro TD absoluto da amostra.
        """
        self.priorities[batch_indices] = np.array(batch_priorities).flatten() + 1e-6

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Agente Deep Q-Network integrando Double DQN, Dueling Architecture e PER.
    Gerencia a política de exploração, o buffer de memória e o ciclo de treinamento.
    """
    def __init__(
        self, q_network, target_network,
        state_space_size, action_space_size,
        learning_rate, discount_factor,
        epsilon_start, epsilon_end, epsilon_decay_steps,
        replay_buffer_size, batch_size, tau, warmup_steps
    ):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.train_step = 0
        self.tau = tau
        self.warmup_steps = warmup_steps
        self.target_network = target_network
        self.q_network = q_network
        self.beta = 0.4
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=replay_buffer_size)

        # Inicialização do otimizador Adam com clipnorm para estabilização de gradientes
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=10.0)

    def choose_action(self, state, env):
        """
        Seleciona uma ação usando a política epsilon-greedy.
        O valor de epsilon decai ao longo do tempo para transitar de exploração para explotação.
        """
        if random.random() < self.epsilon:
            return env.action_space.sample()
        q_values = self._predict_q_values(np.expand_dims(state, axis=0).astype(np.float32))[0]
        return int(np.argmax(q_values))

    def store_experience(self, state, action, reward, next_state, done):
        """Registra a transição no buffer de replay priorizado."""
        self.replay_buffer.store((state, action, reward, next_state, done))

    def update_q_network(self):
        """
        Executa uma etapa de otimização da rede Q.
        Implementa a lógica de Double DQN para o cálculo dos alvos (targets).
        Realiza o Soft Target Update (Polyak Averaging) para sincronização das redes.
        """
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        # Annealing do parâmetro beta do PER
        self.beta = min(1.0, 0.4 + self.train_step / self.epsilon_decay_steps)

        # Amostragem do buffer com prioridade
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size, self.beta)

        # Lógica Double DQN: rede online escolhe ação, rede target avalia valor
        next_q_online = self._predict_q_values(next_states)
        next_actions = tf.argmax(next_q_online, axis=1)
        next_q_target = self._predict_target_q_values(next_states)
        next_action_mask = tf.one_hot(next_actions, depth=self.action_space_size)
        next_q_values = tf.reduce_sum(next_q_target * next_action_mask, axis=1, keepdims=True)

        # Cálculo do alvo de Bellman
        targets = rewards + self.discount_factor * next_q_values.numpy() * (1 - dones)

        # Etapa de gradiente
        loss, td_errors, avg_q = self._train_step(
            states, actions, targets.astype(np.float32), weights
        )

        # Atualização de prioridades no buffer baseada no erro TD
        self.replay_buffer.update_priorities(indices, np.abs(td_errors.numpy()))

        # Atualização do epsilon
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.train_step / self.epsilon_decay_steps
        )
        self.train_step += 1

        # Soft Target Update: theta_target = tau * theta_online + (1 - tau) * theta_target
        q_weights = self.q_network.get_weights()
        target_weights = self.target_network.get_weights()
        new_weights = [
            self.tau * qw + (1 - self.tau) * tw
            for qw, tw in zip(q_weights, target_weights)
        ]
        self.target_network.set_weights(new_weights)

        return float(loss), float(avg_q)

    @tf.function
    def _predict_q_values(self, state):
        """Predição otimizada dos valores Q usando a rede online."""
        return self.q_network(state, training=False)

    @tf.function
    def _predict_target_q_values(self, next_states):
        """Predição otimizada dos valores Q usando a rede target."""
        return self.target_network(next_states, training=False)

    @tf.function
    def _train_step(self, states, actions, targets, weights):
        """
        Executa a etapa de treinamento (backpropagation) no TensorFlow.
        Calcula a Huber Loss ponderada pelos pesos IS e aplica os gradientes.
        """
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, depth=self.action_space_size)
            predicted = tf.reduce_sum(action_masks * q_values, axis=1, keepdims=True)

            td_errors = tf.abs(targets - predicted)

            # Huber loss para estabilidade numérica
            huber = tf.keras.losses.Huber(reduction='none')(targets, predicted)
            loss = tf.reduce_mean(huber * tf.cast(weights, tf.float32))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        return loss, td_errors, tf.reduce_mean(q_values)


# =============================================================================
# INICIALIZAÇÃO E LOOP PRINCIPAL
# =============================================================================
q_network = create_q_network(state_space_size, action_space_size)
target_network = create_q_network(state_space_size, action_space_size)
target_network.set_weights(q_network.get_weights())

agent = DQNAgent(
    q_network, target_network,
    state_space_size, action_space_size,
    learning_rate, discount_factor,
    epsilon_start if not load_existing_weights else epsilon_end,
    epsilon_end, epsilon_decay_steps,
    replay_buffer_size, batch_size, tau,
    warmup_steps
)

if load_existing_weights:
    try:
        # Build das redes com entrada dummy para permitir carregamento de pesos
        dummy = np.zeros((1, state_space_size), dtype=np.float32)
        q_network(dummy)
        target_network(dummy)
        q_network.load_weights(weights_file)
        target_network.load_weights(weights_file)
        print(f"✅ Pesos carregados: {weights_file}")
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️  Não foi possível carregar pesos: {e}. Iniciando do zero.")


num_episodes = 1000
target_score = 475
consecutive_successes = 0
best_reward = 0

for episode in range(num_episodes):
    state, info = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0
    done = False
    episode_losses = []
    episode_qs = []

    while not done:
        action = agent.choose_action(state, env)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        done = done or truncated

        agent.store_experience(state, action, reward, next_state, done)

        result = agent.update_q_network()
        if result:
            loss, avg_q = result
            episode_losses.append(loss)
            episode_qs.append(avg_q)

        state = next_state
        total_reward += reward

    # Registro de métricas no TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('Reward', total_reward, step=episode)
        tf.summary.scalar('Epsilon', agent.epsilon, step=episode)
        if episode_losses:
            tf.summary.scalar('Loss', np.mean(episode_losses), step=episode)
            tf.summary.scalar('Avg_Q_Value', np.mean(episode_qs), step=episode)

    # Checkpoint do melhor modelo baseado na recompensa total
    if total_reward >= best_reward:
        best_reward = total_reward
        q_network.save_weights(weights_file)
        print(f"🏆 Novo recorde! Recompensa: {total_reward:.0f} | Pesos salvos.")

    print(
        f"Ep {episode + 1:4d} | "
        f"Reward: {total_reward:6.1f} | "
        f"ε: {agent.epsilon:.3f} | "
        f"β: {agent.beta:.3f} | "
        f"Buffer: {len(agent.replay_buffer):6d}"
    )

    if total_reward >= target_score:
        consecutive_successes += 1
        if consecutive_successes >= 10:
            print(f"\n✅ Problema resolvido em {episode + 1} episódios!")
            break
    else:
        consecutive_successes = 0

env.close()
