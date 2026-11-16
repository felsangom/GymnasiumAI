"""
Agente DQN (CNN) para o CarRacing-v3. Execute a partir da raiz do repositório:

    python -m GymCarRacing.car_racing            # treina (sem janela)
    python -m GymCarRacing.car_racing --play --render
"""
from pathlib import Path

import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers

from common.buffers import FrameStackReplayBuffer
from common.dqn import DQNAgent
from common.training import TrainConfig, run_cli

from .safedriving_wrapper import FrameStackSkipWrapper, SafeDrivingWrapper

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# HIPERPARÂMETROS
# =============================================================================
FRAME_SKIP = 4
FRAME_STACK = 4
STATE_SHAPE = (96, 96, FRAME_STACK)
# O buffer guarda 2 frames uint8 por transição (~18 KB): 20000 ≈ 370 MB.
MEMORY_SIZE = 20000
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 0.00025
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 10000   # constante de tempo do decaimento (em passos de treino)
WARMUP_STEPS = 1000
TARGET_UPDATE_STEPS = 1000    # cópia completa do target a cada N passos de treino

CONFIG = TrainConfig(
    name="CarRacing-v3",
    weights_file=BASE_DIR / "car_racing_dqn_weights.weights.h5",
    log_dir=BASE_DIR / "logs" / "car_racing",
    num_episodes=1000,
    target_score=900,         # score original do jogo, em avaliação greedy
    eval_interval=25,
    eval_episodes=20,
    reward_window_size=20,
    epsilon_when_loaded=EPSILON_END,
)


def make_env(render_mode=None):
    env = gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
    return FrameStackSkipWrapper(SafeDrivingWrapper(env), FRAME_SKIP, FRAME_STACK)


def build_model(action_size):
    """CNN estilo Atari; a normalização /255 fica dentro do modelo (buffer em uint8)."""
    return tf.keras.Sequential([
        layers.Input(STATE_SHAPE),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
        layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(action_size, activation='linear'),
    ])


def build_agent(env):
    action_size = int(env.action_space.n)
    model = build_model(action_size)
    target_model = build_model(action_size)
    target_model.set_weights(model.get_weights())
    return DQNAgent(
        model, target_model, action_size,
        FrameStackReplayBuffer(MEMORY_SIZE, FRAME_STACK),
        LEARNING_RATE, GAMMA,
        EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS,
        BATCH_SIZE, warmup_steps=WARMUP_STEPS, target_update_interval=TARGET_UPDATE_STEPS,
    )


if __name__ == "__main__":
    run_cli(CONFIG, make_env, build_agent)
