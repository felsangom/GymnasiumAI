import random
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Hiperparâmetros
learning_rate = 0.001
discount_factor = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay_steps = 1000
replay_buffer_size = 10000
batch_size = 64
sync_target_interval = 100
weights_file = "cartpole_dqn_weights.weights.h5"
load_existing_weights = True

# Cria o ambiente CartPole-v1
env = gym.make('CartPole-v1', render_mode="human")
state_space_size = env.observation_space.shape[-1]
action_space_size = env.action_space.n

# Define a arquitetura da rede neural (Q-Network)
def create_q_network(state_space_size, action_space_size):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(state_space_size,)),
        Dense(64, activation='relu'),
        Dense(action_space_size, activation='linear')  # Saída para valores Q
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Cria as redes Q (online e target)
q_network = create_q_network(state_space_size, action_space_size)
target_network = create_q_network(state_space_size, action_space_size)
target_network.set_weights(q_network.get_weights())

# Implementa o agente DQN
class DQNAgent:
    def __init__(self, q_network, target_network, state_space_size, action_space_size, learning_rate, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps, replay_buffer_size, batch_size, sync_target_interval):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.train_step = 0
        self.sync_target_interval = sync_target_interval
        self.target_network = target_network
        self.q_network = q_network

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            q_values = self._predict_q_values(np.expand_dims(state, axis=0))[0]
            return np.argmax(q_values)

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_experience(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(next_states), np.array(dones).reshape(-1, 1)

    def update_q_network(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_experience()

        # Calcula os valores Q alvo usando a rede target
        target_q_values = self._predict_target_q_values(next_states)
        max_next_q_values = tf.reduce_max(target_q_values, axis=1, keepdims=True)
        targets = rewards + self.discount_factor * max_next_q_values * (1 - dones)

        # Realiza a etapa de treinamento otimizada com o @tf.function para melhor desempenho
        self._train_step(states, actions, targets)

        # Atualiza o epsilon
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.train_step / self.epsilon_decay_steps
        )
        self.train_step += 1

        # Sincroniza as redes Q e target
        if self.train_step % self.sync_target_interval == 0:
            self.target_network.set_weights(self.q_network.get_weights())

    @tf.function
    def _predict_q_values(self, state):
        return self.q_network(state)

    @tf.function
    def _train_step(self, states, actions, targets):
        with tf.GradientTape() as tape:
            q_values = self.q_network(states)
            action_masks = tf.one_hot(actions, depth=self.action_space_size)
            predicted_q_values = tf.reduce_sum(action_masks * q_values, axis=1, keepdims=True)
            loss = tf.keras.losses.mse(targets, predicted_q_values)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        return loss

    @tf.function
    def _predict_target_q_values(self, next_states):
        return self.target_network(next_states)


if load_existing_weights:
    epsilon_start = epsilon_end

# Inicializa o agente
agent = DQNAgent(q_network, target_network, state_space_size, action_space_size, learning_rate, discount_factor, epsilon_start, epsilon_end, epsilon_decay_steps, replay_buffer_size, batch_size, sync_target_interval)
# Adiciona o otimizador ao agente
agent.q_optimizer = Adam(learning_rate=learning_rate)

# Carrega pesos se solicitado
if load_existing_weights:
    try:
        q_network.load_weights(weights_file)
        target_network.load_weights(weights_file)
        print(f"Pesos carregados do arquivo: {weights_file}")
    except FileNotFoundError:
        print("Arquivo de pesos não encontrado. Iniciando treinamento do zero.")

# Loop de treinamento
num_episodes = 500
target_score = 195
consecutive_successes = 0

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.update_q_network()
        state = next_state
        total_reward += reward

    print(f"Episódio: {episode + 1}, Recompensa Total: {total_reward:.2f}, Épsilon: {agent.epsilon:.2f}")

    if total_reward >= target_score:
        consecutive_successes += 1
        if consecutive_successes >= 10:
            print(f"\nProblema resolvido em {episode + 1} episódios!")
            break
    else:
        consecutive_successes = 0

# Salva os pesos após o treinamento
q_network.save_weights(weights_file)
print(f"Pesos da rede Q salvos em: {weights_file}")

env.close()
