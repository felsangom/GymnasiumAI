import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402

from common.dqn import (  # noqa: E402
    DQNAgent, PrioritizedReplayBuffer, create_q_network, evaluate,
    get_render_mode, set_seed, train, try_load_weights,
)

BASE_DIR = Path(__file__).resolve().parent

# =============================================================================
# HIPERPARÂMETROS
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
weights_file = BASE_DIR / "cartpole_dqn_weights.weights.h5"
log_dir = BASE_DIR / "logs" / "dqn"
load_existing_weights = False
seed = 42

num_episodes = 1000
target_score = 475          # média móvel necessária para considerar resolvido
reward_window_size = 10     # janela da média móvel (e do checkpoint)
eval_episodes = 10          # avaliação greedy ao final (0 desativa)


def main():
    set_seed(seed)
    # Use RENDER=human para ver a janela; por padrão não renderiza.
    env = gym.make('CartPole-v1', render_mode=get_render_mode())
    state_space_size = env.observation_space.shape[-1]
    action_space_size = int(env.action_space.n)

    q_network = create_q_network(state_space_size, action_space_size)
    target_network = create_q_network(state_space_size, action_space_size)
    target_network.set_weights(q_network.get_weights())

    loaded = load_existing_weights and try_load_weights(
        [q_network, target_network], str(weights_file)
    )

    agent = DQNAgent(
        q_network, target_network, action_space_size,
        PrioritizedReplayBuffer(replay_buffer_size),
        learning_rate, discount_factor,
        epsilon_end if loaded else epsilon_start, epsilon_end, epsilon_decay_steps,
        batch_size, tau=tau, warmup_steps=warmup_steps,
    )

    try:
        train(
            env, agent, num_episodes=num_episodes, log_dir=log_dir,
            weights_file=str(weights_file), reward_window_size=reward_window_size,
            target_mean=target_score, consecutive_required=1, seed=seed,
        )
        if eval_episodes > 0:
            try_load_weights([q_network], str(weights_file))
            evaluate(env, agent, eval_episodes, seed=seed)
    finally:
        env.close()


if __name__ == "__main__":
    main()
