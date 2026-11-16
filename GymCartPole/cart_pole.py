"""
Agente DQN para o CartPole-v1. Execute a partir da raiz do repositório:

    python -m GymCartPole.cart_pole            # treina (sem janela)
    python -m GymCartPole.cart_pole --play --render
"""
from pathlib import Path

import gymnasium as gym

from common.buffers import PrioritizedReplayBuffer
from common.dqn import DQNAgent, create_q_network
from common.training import TrainConfig, run_cli

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# HIPERPARÂMETROS
# =============================================================================
LEARNING_RATE = 0.0005
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 5000
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAU = 0.01                  # soft target update (Polyak averaging)
WARMUP_STEPS = 1000

CONFIG = TrainConfig(
    name="CartPole-v1",
    weights_file=BASE_DIR / "cartpole_dqn_weights.weights.h5",
    log_dir=BASE_DIR / "logs" / "dqn",
    num_episodes=1000,
    target_score=475,       # score médio em avaliação greedy
    eval_interval=10,
    eval_episodes=10,
    epsilon_when_loaded=EPSILON_END,
)


def make_env(render_mode=None):
    return gym.make('CartPole-v1', render_mode=render_mode)


def build_agent(env):
    state_space_size = env.observation_space.shape[-1]
    action_space_size = int(env.action_space.n)
    q_network = create_q_network(state_space_size, action_space_size)
    target_network = create_q_network(state_space_size, action_space_size)
    target_network.set_weights(q_network.get_weights())
    return DQNAgent(
        q_network, target_network, action_space_size,
        PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE),
        LEARNING_RATE, DISCOUNT_FACTOR,
        EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS,
        BATCH_SIZE, tau=TAU, warmup_steps=WARMUP_STEPS,
    )


if __name__ == "__main__":
    run_cli(CONFIG, make_env, build_agent)
