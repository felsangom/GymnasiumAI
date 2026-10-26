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
- Prioritized Experience Replay (PER)
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

- Deep Q-Network with CNN-based state processing
- Experience Replay
- Target Network
- Epsilon-greedy exploration
- Image preprocessing and grayscale conversion
- Custom reward shaping based on vehicle speed and track position
- Custom environment wrapper for detecting when the vehicle leaves the track
- Pre-trained model weights

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

A recorded example of the trained agent is also available:

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
pip install numpy tensorflow shapely "gymnasium[all]"
```

## Running the Agents

Each environment has its own implementation.

### CartPole

```bash
cd GymCartPole
python cart_pole.py
```

### LunarLander

```bash
cd GymLunarLander
python lunar_lander.py
```

### CarRacing

```bash
cd GymCarRacing
python car_racing.py
```

Some directories include pre-trained weights that can be used to run or continue training existing agents.

Training reinforcement learning agents can require a significant number of interactions with the environment, and results may vary between executions because exploration and environment behavior are stochastic.

## Project Structure

```text
GymnasiumAI/
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
└── README.md
```

## About

This repository is part of my exploration of Artificial Intelligence, Machine Learning and Reinforcement Learning.

Rather than relying only on high-level reinforcement learning libraries, these implementations explore the underlying mechanics of DQN-based agents, including replay buffers, target networks, exploration strategies, neural network architectures and training loops.
