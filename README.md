# GymnasiumAI

Reinforcement Learning experiments using Python, TensorFlow and Gymnasium.

This repository contains implementations of Deep Reinforcement Learning agents for different Gymnasium environments, exploring techniques such as Deep Q-Networks, Double DQN, Dueling Networks, Prioritized Experience Replay and reward shaping.

The goal of this project is to explore how different reinforcement learning techniques behave across environments with different state and action spaces — from small numerical state vectors to image-based input.

## Environments

### CartPole

A DQN-based agent for the `CartPole-v1` environment.

The implementation combines several techniques intended to improve training stability and convergence:

- Deep Q-Network (DQN)
- Double DQN
- Dueling DQN architecture
- Prioritized Experience Replay (PER, Sum Tree)
- Epsilon-greedy exploration with decay
- Soft target network updates using Polyak averaging
- Gradient clipping
- TensorBoard training metrics

The Dueling architecture separates the estimation of the state value from the advantage of each possible action before combining them into the final Q-values.

Directory:

```text
GymCartPole/
```

Main implementation:

```text
GymCartPole/cart_pole.py
```

---

### LunarLander

A reinforcement learning agent for the `LunarLander-v3` environment.

This implementation also uses an enhanced DQN architecture designed to improve learning stability:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Importance Sampling correction
- Epsilon-greedy exploration
- Soft target updates
- Experience replay warm-up
- Gradient clipping
- TensorBoard monitoring

The agent learns how to control the lander by estimating the expected long-term reward of each available action.

Directory:

```text
GymLunarLander/
```

Main implementation:

```text
GymLunarLander/lunar_lander.py
```

---

### CarRacing

An image-based DQN agent for the `CarRacing-v3` environment.

Unlike CartPole and LunarLander, the state is obtained directly from rendered frames. The agent therefore uses a Convolutional Neural Network to extract relevant features from the environment.

The implementation includes:

- Double DQN with CNN-based state processing
- Experience Replay storing single uint8 frames and rebuilding the stacks at sampling time (4x less memory)
- Hard-updated Target Network
- Epsilon-greedy exploration
- Grayscale conversion, frame skip and frame stacking (4 frames, gives the agent a notion of motion)
- Light reward shaping (small speed bonus, off-track penalty); the original game reward is tracked separately and is the score used for evaluation and checkpoints
- Custom environment wrapper for detecting when the vehicle leaves the track (uniform safety margin around the road)

Pre-trained weights are included (`car_racing_dqn_weights.weights.h5`), trained for 700 episodes on the stacked-frame model (greedy score of about 910 on the 20 fixed evaluation tracks).

The environment was extended with track boundary detection so episodes can be terminated when the vehicle moves too far outside the track. This reduces time spent exploring clearly invalid states and makes training more focused.

Directory:

```text
GymCarRacing/
```

Main implementation:

```text
GymCarRacing/car_racing.py
```

Custom environment wrapper:

```text
GymCarRacing/safedriving_wrapper.py
```

A recorded example of a trained agent is also available (recorded with an earlier, single-frame version of the model):

[CarRacing demo](GymCarRacing/car_racing.mp4)

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Gymnasium
- Shapely
- TensorBoard

## Reinforcement Learning Techniques

Some of the techniques explored in this repository include:

### Deep Q-Network

A neural network approximates the action-value function:

```text
Q(s, a)
```

allowing Q-Learning to be applied to environments where maintaining a traditional Q-table would not be practical.

### Double DQN

Double DQN reduces the tendency of traditional DQN to overestimate action values.

The online network selects the next action while the target network evaluates its expected value.

### Dueling DQN

The network independently estimates:

```text
V(s)
```

the value of a state, and:

```text
A(s, a)
```

the advantage of performing each action.

They are then combined to estimate:

```text
Q(s, a)
```

This allows the agent to learn the value of states even when the choice between actions has little impact.

### Prioritized Experience Replay

Instead of sampling every stored experience with equal probability, transitions with larger Temporal Difference errors receive higher priority.

This allows the agent to revisit experiences from which it currently has more to learn.

Importance Sampling weights are used during training to compensate for the bias introduced by prioritized sampling.

### Target Networks

Separate target networks are used to make Q-Learning targets more stable.

CartPole and LunarLander use gradual target synchronization through Polyak averaging, while the CarRacing implementation periodically synchronizes the target network.

## Installation

Clone the repository:

```bash
git clone https://github.com/felsangom/GymnasiumAI.git
cd GymnasiumAI
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Agents

Run every agent as a module **from the repository root**. By default no window is opened (faster training, works on servers).

```bash
python -m GymCartPole.cart_pole
python -m GymLunarLander.lunar_lander
python -m GymCarRacing.car_racing
```

Options (the same for the three agents):

| Option | Description |
|---|---|
| `--play [N]` | Do not train: load the saved weights and play `N` greedy episodes (default 5) |
| `--render` | Open the environment window (same as `RENDER=human`) |
| `--episodes N` | Maximum number of training episodes |
| `--load` | Start training from the saved weights |
| `--resume` | Resume training from the last saved training state |
| `--seed N` | Random seed |
| `--target-score X` | Greedy evaluation score that ends training (overrides the environment default) |

Watching a trained agent:

```bash
python -m GymLunarLander.lunar_lander --play --render
```

How training works:

- Every few episodes the agent is evaluated greedily (no exploration) on a separate environment with fixed seeds. Checkpoints and the "solved" criterion use this score, not the noisy training episodes.
- Saved weights are only replaced when a new evaluation beats the score recorded in the `.meta.json` file next to them, so starting a new training run never destroys a better model.
- The full training state (networks, optimizer, step counters) is saved at every evaluation and on `Ctrl+C`; continue with `--resume`. The replay buffer is not saved and is refilled during warm-up.
- TensorBoard logs are written to `logs/` next to each script (git-ignored): `tensorboard --logdir GymCartPole/logs`.

Training reinforcement learning agents can require a significant number of interactions with the environment, and results may vary between executions because exploration and environment behavior are stochastic.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project Structure

```text
GymnasiumAI/
├── common/
│   ├── buffers.py      # uniform, prioritized (Sum Tree) and frame-stack replay buffers
│   ├── dqn.py          # Dueling network and DQN agent
│   └── training.py     # training loop, greedy evaluation and command line
│
├── GymCartPole/
│   ├── cart_pole.py
│   └── ...
│
├── GymLunarLander/
│   ├── lunar_lander.py
│   └── ...
│
├── GymCarRacing/
│   ├── car_racing.py
│   ├── safedriving_wrapper.py
│   ├── car_racing.mp4
│   └── ...
│
├── tests/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## About

This repository is part of my exploration of Artificial Intelligence, Machine Learning and Reinforcement Learning.

Rather than relying only on high-level reinforcement learning libraries, these implementations explore the underlying mechanics of DQN-based agents, including replay buffers, target networks, exploration strategies, neural network architectures and training loops.
