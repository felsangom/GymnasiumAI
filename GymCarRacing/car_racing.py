import random
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from safedriving_wrapper import SafeDrivingWrapper
from tensorflow.keras import layers

# Configuração do ambiente
env_raw = gym.make("CarRacing-v3", continuous=False, render_mode="human")
env = SafeDrivingWrapper(env_raw)

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

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA)

    def _build_model(self):
        """Builds the CNN model."""
        model = tf.keras.Sequential([
            layers.Input(STATE_SHAPE),
            layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
            layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
            layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(ACTION_SIZE, activation='linear') # Q-values
        ])
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.predict(self.model, tf.expand_dims(state, axis=0))
        return np.argmax(q_values[0].numpy())

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Trains the model using a random batch of experiences from memory."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        # Predict Q-values for current states and next states
        q_values_current = self.predict(self.model, states).numpy()
        q_values_next = self.predict(self.target_model, next_states).numpy()

        for i in range(BATCH_SIZE):
            if dones[i]:
                q_values_current[i, actions[i]] = rewards[i]
            else:
                q_values_current[i, actions[i]] = rewards[i] + GAMMA * np.amax(q_values_next[i])

        self.train_step(states, q_values_current)

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def save_weights(self):
        if SAVE_WEIGHTS:
            self.model.save_weights(WEIGHTS_FILE)

    def load_weights(self):
        if LOAD_EXISTING_WEIGHTS:
            try:
                self.model.load_weights(WEIGHTS_FILE)
                self.update_target_model()
                self.epsilon = EPSILON_MIN
                print("Pre-trained weights found. Resuming training...")
            except FileNotFoundError:
                print("No pre-trained weights found. Starting from scratch...")

    @tf.function
    def predict(self, model, states):
        return model(states, training=False)

    @tf.function
    def train_step(self, states, targets):
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            loss = tf.keras.losses.MeanSquaredError()(targets, q_values)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


def preprocess_state(state):
    """Converts the image to grayscale, resizes, and normalizes it."""
    # [Nota do revisor: A conversão para TF tensor é mais eficiente]
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.image.rgb_to_grayscale(state_tensor)
    return tf.cast(state_tensor, tf.float32) / 255.0

def train_agent(episodes=1000):
    agent = DQNAgent()
    agent.load_weights()

    for episode in range(episodes):
        state, _ = env.reset()
        state = preprocess_state(state)

        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = preprocess_state(next_state)

            # Custom Reward Shaping
            car_on_track = env.car_on_track()
            if car_on_track:
                speed = np.linalg.norm(env.unwrapped.car.hull.linearVelocity)
                # Reward for high speed, penalize for low speed
                speed_bonus = max(0, speed * 0.1)
                low_speed_penalty = -1 if speed < 1.0 else 0
                reward += speed_bonus + low_speed_penalty
            else:
                # This penalty is now handled by the wrapper, but we can add more
                reward -= 2

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_model()

        if episode > 0 and episode % SAVE_WEIGHTS_INTERVAL == 0:
            agent.save_weights()

        print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

# Let's train!
train_agent()
