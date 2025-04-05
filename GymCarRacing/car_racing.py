import random
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Configuração do ambiente
env = gym.make("CarRacing-v3", continuous=False, render_mode="human")

# Hiperparâmetros
STATE_SHAPE = (96, 96, 1)
ACTION_SIZE = env.action_space.n
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
ALPHA = 0.00025
EPSILON = 1.0
EPSILON_DECAY = 0.99999
EPSILON_MIN = 0.01
TARGET_UPDATE = 25
WEIGHTS_FILE = "car_racing.weights.h5"
SAVE_WEIGHTS = False
LOAD_EXISTING_WEIGHTS = True
SAVE_WEIGHTS_INTERVAL = 100

def preprocess_state(state):
    """ Converte a imagem para escala de cinza e normaliza."""
    state = tf.image.rgb_to_grayscale(state)
    state = tf.image.resize(state, (96, 96))
    return state / 255.0

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

    def build_model(self):
        """ Cria a rede neural."""
        model = tf.keras.Sequential([
            layers.Input(STATE_SHAPE),
            layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
            layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
            layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(ACTION_SIZE, activation='linear')
        ])
        return model

    def update_target_model(self):
        """ Copia os pesos do modelo principal para o modelo alvo."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """ Escolhe uma ação baseada na política epsilon-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.predict(self.model, tf.expand_dims(state, axis=0))
        return np.argmax(q_values[0].numpy())

    def remember(self, state, action, reward, next_state, done):
        """ Armazena a experiência no replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """ Treina o modelo com amostras da memória."""
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)

        q_values = self.predict(self.model, states).numpy()
        next_q_values = self.predict(self.target_model, next_states).numpy()

        for i in range(BATCH_SIZE):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + GAMMA * np.amax(next_q_values[i])

        self.train_step(states, q_values)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def save_weights(self):
        if SAVE_WEIGHTS:
            self.model.save_weights(WEIGHTS_FILE)

    def load_weights(self):
        if LOAD_EXISTING_WEIGHTS:
            try:
                self.model.load_weights(WEIGHTS_FILE)
                self.target_model.load_weights(WEIGHTS_FILE)
                self.epsilon = EPSILON_MIN
                print("Arquivo de pré-treino encontrado, continuando treinamento...")
            except FileNotFoundError:
                print("Arquivo de pré-treino não encontrado, iniciando treinamento...")

    @tf.function
    def predict(self, model, states):
        """ Faz a previsão dos valores Q para múltiplos estados."""
        return model(states, training=False)

    @tf.function
    def train_step(self, states, targets):
        """ Passo de treinamento otimizado com @tf.function."""
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            loss = tf.keras.losses.MSE(targets, q_values)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

# Treinamento
def train_agent(episodes=1000):
    agent = DQNAgent()
    agent.load_weights()

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)
        real_reward = 0
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_state(next_state)

            real_reward += reward

            # Modificação da recompensa
            car_on_track = env.unwrapped.car_on_track()
            if car_on_track:
                speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
                speed_bonus = speed * 0.1
                low_speed_penalty = -2 if speed < 2 else 0
                reward += speed_bonus + low_speed_penalty
            else:
                reward -= 2

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_model()

        if episode > 0 and episode % SAVE_WEIGHTS_INTERVAL == 0:
            agent.save_weights()

        print(f"Episódio {episode+1}, Recompensa: {total_reward}, Recompensa real: {real_reward}, Epsilon: {agent.epsilon:.2f}")

train_agent()
