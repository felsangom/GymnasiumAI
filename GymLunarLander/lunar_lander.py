import datetime
import random
import os
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# =============================================================================
# CONFIGURAÇÕES E HIPERPARÂMETROS
# =============================================================================
# Taxa de aprendizado para o otimizador Adam
learning_rate = 0.0005
# Fator de desconto para recompensas futuras (Bellman Equation)
discount_factor = 0.99
# Parâmetros da política Epsilon-Greedy (Exploração vs. Explotação)
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_steps = 20000
# Configurações do Replay Buffer
replay_buffer_size = 100000
batch_size = 64
# Parâmetro para Soft Target Update (Polyak Averaging)
# Valor ajustado para 0.01 para maior estabilidade em ambientes de dinâmica rápida
tau = 0.01
# Número de passos iniciais antes de começar o treinamento (coleta de dados)
warmup_steps = 5000
# Janela para cálculo da média móvel de recompensas
reward_window_size = 100

weights_file = "lunarlander_dqn_weights.weights.h5"
load_existing_weights = False

# Configuração do TensorBoard para monitoramento de métricas
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/lunar_lander/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

# Inicialização do ambiente LunarLander-v3
env = gym.make('LunarLander-v3', render_mode="human")
state_space_size = env.observation_space.shape[-1]
action_space_size = env.action_space.n


# =============================================================================
# ARQUITETURA DA REDE NEURAL (DUELING DQN)
# =============================================================================
class DuelingAggregation(tf.keras.layers.Layer):
    """
    Camada customizada para agregar os ramos de Valor (V) e Vantagem (A).
    A fórmula utilizada é: Q(s,a) = V(s) + (A(s,a) - média(A(s,a))).
    A subtração da média garante que a vantagem seja centrada, melhorando a estabilidade.
    """
    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))


def create_q_network(state_space_size, action_space_size):
    """
    Constrói uma Dueling Q-Network usando a API Funcional do Keras.
    Esta arquitetura permite ao agente aprender quais estados são valiosos
    independentemente das ações disponíveis.
    """
    inputs = Input(shape=(state_space_size,))
    
    # Camadas compartilhadas para extração de características do estado
    shared = Dense(256, activation='relu')(inputs)
    shared = Dense(256, activation='relu')(shared)

    # Ramo de Valor do Estado (State Value - V)
    value = Dense(128, activation='relu')(shared)
    value = Dense(1, activation='linear')(value)

    # Ramo de Vantagem da Ação (Action Advantage - A)
    advantage = Dense(128, activation='relu')(shared)
    advantage = Dense(action_space_size, activation='linear')(advantage)

    # Agregação final para obter os valores Q
    outputs = DuelingAggregation()([value, advantage])
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


# =============================================================================
# BUFFER DE REPLAY PRIORIZADO (PER)
# =============================================================================
class PrioritizedReplayBuffer:
    """
    Implementação de Prioritized Experience Replay.
    Permite que o agente aprenda mais frequentemente com experiências que resultam
    em erros de predição (TD Error) maiores, acelerando a convergência.
    """
    def __init__(self, maxlen, alpha=0.4):
        self.maxlen = maxlen
        self.alpha = alpha  # Controla o nível de priorização (0 = uniforme, 1 = total)
        self.buffer = []
        self.priorities = np.zeros((maxlen,), dtype=np.float32)
        self.pos = 0

    def store(self, experience):
        """Armazena a experiência com prioridade máxima inicial."""
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.maxlen:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.maxlen

    def sample(self, batch_size, beta=0.4):
        """Realiza a amostragem baseada em probabilidade e calcula pesos de Importance Sampling."""
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Pesos para corrigir o viés introduzido pela amostragem não-uniforme
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32).reshape(-1, 1),
            np.array(weights, dtype=np.float32),
            indices
        )

    def update_priorities(self, batch_indices, batch_priorities):
        """Atualiza as prioridades no buffer após o passo de treino."""
        self.priorities[batch_indices] = np.array(batch_priorities).flatten() + 1e-6

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# AGENTE DEEP Q-NETWORK (DQN)
# =============================================================================
class DQNAgent:
    """
    Agente que integra Double DQN, Arquitetura Dueling e Prioritized Replay.
    Implementa a lógica de decisão, armazenamento de memória e otimização.
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
        self.beta = 0.4  # Parâmetro para Importance Sampling no PER
        self.replay_buffer = PrioritizedReplayBuffer(maxlen=replay_buffer_size)

        # Otimizador com clipnorm para evitar explosão de gradientes devido às recompensas do LunarLander
        self.optimizer = Adam(learning_rate=learning_rate, clipnorm=10.0)

    def choose_action(self, state, env):
        """Seleciona uma ação usando a política epsilon-greedy."""
        if random.random() < self.epsilon:
            return env.action_space.sample()
        q_values = self._predict_q_values(
            np.expand_dims(state, axis=0).astype(np.float32)
        )[0]
        return int(np.argmax(q_values))

    def store_experience(self, state, action, reward, next_state, done):
        """Registra a transição no Replay Buffer."""
        self.replay_buffer.store((state, action, reward, next_state, done))

    def update_q_network(self):
        """Executa uma iteração de treinamento (aprendizado do agente)."""
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        # Incrementa linearmente o beta para Importance Sampling
        self.beta = min(1.0, 0.4 + self.train_step / self.epsilon_decay_steps)

        # Amostragem do buffer
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size, self.beta)

        # Lógica Double DQN: Rede online escolhe a ação, rede target avalia o valor
        next_q_online = self._predict_q_values(next_states)
        next_actions = tf.argmax(next_q_online, axis=1)
        next_q_target = self._predict_target_q_values(next_states)
        next_action_mask = tf.one_hot(next_actions, depth=self.action_space_size)
        next_q_values = tf.reduce_sum(
            next_q_target * next_action_mask, axis=1, keepdims=True
        )

        # Cálculo do alvo de Bellman
        targets = rewards + self.discount_factor * next_q_values.numpy() * (1 - dones)

        # Passo de otimização (Backpropagation)
        loss, td_errors, avg_q = self._train_step(
            states, actions, targets.astype(np.float32), weights
        )

        # Atualiza prioridades no buffer baseada no erro TD absoluto
        self.replay_buffer.update_priorities(indices, np.abs(td_errors.numpy()))

        # Decaimento do Epsilon (Exploração)
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.train_step / self.epsilon_decay_steps
        )
        self.train_step += 1
        
        # Soft Update das redes (alinhamento gradual)
        self._soft_update()

        return float(loss), float(avg_q)

    def _soft_update(self):
        """Sincronização suave entre a rede Q (online) e a rede Target."""
        q_w = self.q_network.get_weights()
        t_w = self.target_network.get_weights()
        self.target_network.set_weights([
            self.tau * qw + (1 - self.tau) * tw for qw, tw in zip(q_w, t_w)
        ])

    @tf.function
    def _predict_q_values(self, state):
        """Inferência otimizada com Grafo do TF para a rede online."""
        return self.q_network(state, training=False)

    @tf.function
    def _predict_target_q_values(self, next_states):
        """Inferência otimizada com Grafo do TF para a rede target."""
        return self.target_network(next_states, training=False)

    @tf.function
    def _train_step(self, states, actions, targets, weights):
        """Passo de treino compilado para máxima performance na GPU/CPU."""
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            action_masks = tf.one_hot(actions, depth=self.action_space_size)
            predicted = tf.reduce_sum(action_masks * q_values, axis=1, keepdims=True)

            td_errors = tf.abs(targets - predicted)
            # Huber Loss: quadrática para erros pequenos, linear para erros grandes (robusta)
            huber = tf.keras.losses.Huber(reduction='none')(targets, predicted)
            loss = tf.reduce_mean(huber * tf.cast(weights, tf.float32))

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        return loss, td_errors, tf.reduce_mean(q_values)


# =============================================================================
# EXECUÇÃO E LOOP PRINCIPAL
# =============================================================================
# Inicialização das redes neurais
q_network = create_q_network(state_space_size, action_space_size)
target_network = create_q_network(state_space_size, action_space_size)
target_network.set_weights(q_network.get_weights())

# Configuração de Epsilon inicial caso os pesos sejam carregados
epsilon_load = 0.3 if load_existing_weights else epsilon_start

# Instanciação do Agente
agent = DQNAgent(
    q_network, target_network,
    state_space_size, action_space_size,
    learning_rate, discount_factor,
    epsilon_load, epsilon_end, epsilon_decay_steps,
    replay_buffer_size, batch_size, tau, warmup_steps
)

# Carregamento de pesos pré-treinados
if load_existing_weights and os.path.exists(weights_file):
    try:
        # Chamada dummy para construir o grafo do modelo antes de carregar pesos
        dummy = np.zeros((1, state_space_size), dtype=np.float32)
        q_network(dummy)
        target_network(dummy)
        q_network.load_weights(weights_file)
        target_network.load_weights(weights_file)
        print(f"🚀 Pesos carregados: {weights_file}")
    except Exception as e:
        print(f"⚠️ Erro ao carregar pesos: {e}. Começando do zero.")


num_episodes = 5000
target_score = 200
consecutive_successes = 0
best_mean_reward = -np.inf

# Buffer para cálculo da média móvel de recompensa dos últimos 100 episódios
reward_window = deque(maxlen=reward_window_size)

try:
    for episode in range(num_episodes):
        state, info = env.reset()
        state = np.array(state, dtype=np.float32)
        total_reward = 0.0
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

        reward_window.append(total_reward)
        mean_reward = np.mean(reward_window)

        # Registro de métricas no TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('Reward', total_reward, step=episode)
            tf.summary.scalar('Mean_Reward_100ep', mean_reward, step=episode)
            tf.summary.scalar('Epsilon', agent.epsilon, step=episode)
            if episode_losses:
                tf.summary.scalar('Loss', np.mean(episode_losses), step=episode)
                tf.summary.scalar('Avg_Q_Value', np.mean(episode_qs), step=episode)

        # Salvamento automático baseado na melhora da média móvel
        if len(reward_window) == reward_window_size and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            q_network.save_weights(weights_file)
            print(f"🏆 Novo melhor modelo: média={mean_reward:.1f} | Pesos salvos.")

        print(
            f"Ep {episode + 1:5d} | "
            f"Reward: {total_reward:8.1f} | "
            f"Média(100): {mean_reward:7.1f} | "
            f"ε: {agent.epsilon:.3f} | "
            f"Buffer: {len(agent.replay_buffer):6d}"
        )

        # Critério de parada: Estabilidade da média acima do alvo por 5 episódios consecutivos
        if len(reward_window) == reward_window_size and mean_reward >= target_score:
            consecutive_successes += 1
            if consecutive_successes >= 5:
                print(f"\n✅ Módulo Lunar pousado com sucesso em {episode + 1} episódios!")
                print(f"   Média final dos últimos 100 episódios: {mean_reward:.1f}")
                break
        else:
            consecutive_successes = 0

finally:
    env.close()
    print("Processo finalizado.")
