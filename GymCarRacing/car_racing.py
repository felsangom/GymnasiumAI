import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
from safedriving_wrapper import FrameStackSkipWrapper, SafeDrivingWrapper  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
import tensorflow as tf  # noqa: E402

from common.dqn import (  # noqa: E402
    DQNAgent, ReplayBuffer, evaluate, get_render_mode, set_seed, train,
    try_load_weights,
)

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# HIPERPARÂMETROS
# =============================================================================
FRAME_SKIP = 4
FRAME_STACK = 4
STATE_SHAPE = (96, 96, FRAME_STACK)
# Estados em uint8 (4 frames) ocupam ~74 KB; com state+next_state, 5000 ≈ 370 MB.
MEMORY_SIZE = 5000
BATCH_SIZE = 32
GAMMA = 0.99
ALPHA = 0.00025
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_STEPS = 10000   # constante de tempo do decaimento exponencial (em passos de treino)
WARMUP_STEPS = 1000
TARGET_UPDATE_STEPS = 1000    # cópia completa do target a cada N passos de treino
REWARD_CLIP = (-10.0, 10.0)   # aplicado só à recompensa usada no treino
EPISODES = 1000
REWARD_WINDOW = 20
SEED = 42
EVAL_EPISODES = 3

# Arquivo novo: os pesos antigos (1 frame) não são compatíveis com a entrada empilhada.
WEIGHTS_FILE = BASE_DIR / "car_racing_stack.weights.h5"
LOG_DIR = BASE_DIR / "logs" / "car_racing"
SAVE_WEIGHTS = True
LOAD_EXISTING_WEIGHTS = False
EPSILON_WHEN_LOADED = EPSILON_MIN


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


def main():
    set_seed(SEED)
    # Use RENDER=human para ver a janela; por padrão não renderiza.
    env_raw = gym.make("CarRacing-v3", continuous=False, render_mode=get_render_mode())
    env = FrameStackSkipWrapper(SafeDrivingWrapper(env_raw), FRAME_SKIP, FRAME_STACK)
    action_size = int(env.action_space.n)

    model = build_model(action_size)
    target_model = build_model(action_size)
    target_model.set_weights(model.get_weights())

    loaded = LOAD_EXISTING_WEIGHTS and try_load_weights(
        [model, target_model], str(WEIGHTS_FILE)
    )

    agent = DQNAgent(
        model, target_model, action_size, ReplayBuffer(MEMORY_SIZE),
        ALPHA, GAMMA,
        EPSILON_WHEN_LOADED if loaded else EPSILON_START, EPSILON_MIN, EPSILON_DECAY_STEPS,
        BATCH_SIZE, warmup_steps=WARMUP_STEPS, target_update_interval=TARGET_UPDATE_STEPS,
    )

    try:
        train(
            env, agent, num_episodes=EPISODES, log_dir=LOG_DIR,
            weights_file=str(WEIGHTS_FILE) if SAVE_WEIGHTS else None,
            reward_window_size=REWARD_WINDOW, reward_clip=REWARD_CLIP, seed=SEED,
        )
        if EVAL_EPISODES > 0:
            if SAVE_WEIGHTS:
                try_load_weights([model], str(WEIGHTS_FILE))
            evaluate(env, agent, EVAL_EPISODES, seed=SEED)
    finally:
        env.close()


if __name__ == "__main__":
    main()
