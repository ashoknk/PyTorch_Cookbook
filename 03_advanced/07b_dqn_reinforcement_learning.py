"""
20. Deep Q-Network (DQN) Reinforcement Learning in PyTorch


This script implements a complete Deep Q-Network (DQN) reinforcement learning 
agent to solve classic control tasks (e.g., CartPole-v1) from the Gymnasium suite. 
It covers Q-learning theory, experience replay buffers to break temporal transitions 
correlation, and target networks to stabilize optimization updates.

The main purpose of this code is to train a Deep Q-Network (DQN) AI agent to play the CartPole game without human intervention.
Instead of writing hardcoded rules (e.g., "if pole leans right, move right"), the code gives the neural network complete control. 
The network starts by making random guesses (exploring), records its experiences into memory, and gradually learns 
which moves keep the pole balanced to achieve the highest score.   


Learning Objectives:
1. Define a Q-Network mapping environment states to action values.
2. Build an Experience Replay Buffer containing transition trajectories.
3. Apply epsilon-greedy strategies to balance environment exploration and exploitation.
4. Set up stable Q-learning target computations using a target network.
"""

import random
from collections import deque
import numpy as np

# We import the core torch library.
import torch

# nn contains the activation and fully connected linear layers.
import torch.nn as nn

# optim contains optimization algorithms (Adam).
import torch.optim as optim

# We import gymnasium for RL environment interactions.
# Documentation: https://gymnasium.farama.org/
import gymnasium as gym

# ==========================================
# 1. DEFINE Q-NETWORK ARCHITECTURE
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # A simple Feedforward Network (MLP) representing action-value approximations
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        # Maps state observations to Q-values for each discrete action
        return self.fc(state)

# ==========================================
# 2. DEFINE REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        # We use a deque with maximum capacity to automatically discard old transitions
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample mini-batches(64) from buffer to break temporal correlations between successive steps. k unique elements
        transitions = random.sample(self.buffer, batch_size)
        
        # Unpack, convert to numpy, and then to PyTorch tensors
        # NOTE: without numpy you will get a "TypeError: expected np.ndarray (got list)" error when creating torch tensors
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

def main():
    print("--- 1. Creating Headless CartPole-v1 Environment ---")
    # Initialize the classic CartPole pole-balancing control game.
    # We specify render_mode=None for headless verification.
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Observation State Space size: {state_dim}, Action Space size: {action_dim}")

    # Initialize Active and Target networks.
    # The target network provides static reference coordinates for temporal differences (TD) loss.
    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    # Align target weights initially. Copies all weights and biases from policy_net directly into target_net.
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    replay_buffer = ReplayBuffer(capacity=5000)

    # RL Hyperparameters
    gamma = 0.99          # Discount factor for future rewards
    epsilon = 1.0         # Starting exploration probability
    epsilon_min = 0.05    # Floor exploration probability
    epsilon_decay = 0.99  # Decay factor per episode
    batch_size = 64
    target_sync_steps = 100
    global_steps = 0

    # ==========================================
    # 3. ENVIRONMENT INTERACTION & TRAINING LOOP
    # ==========================================
    episodes = 25
    print(f"--- Running DQN agent for {episodes} episodes ---")
    
    for episode in range(episodes): # manages the overall game of CartPole-v1 
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done: # runs step-by-step frame during a single match until the pole falls
            global_steps += 1
            
            # 3.1 Epsilon-Greedy Action Selection
            # random.random() produces a decimal between 0.0 and 1.0
            if random.random() < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else: 
                # (Exploitation) uses the Neural Network's best calculated guess. The AI will consult its trained policy_net neural network to choose the smartest move.
                # Calculate the model's smartest decision based on its current trained weights.
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = torch.argmax(policy_net(state_t), dim=1).item()

            # Execute step in Gymnasium
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition in Replay Memory
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # 3.2 Optimization Step
            # Checks if there are enough saved game steps inside the ReplayBuffer to fill a full mini-batch (e.g., 64 past steps).
            # For the first few steps of Episode 1, the buffer is empty. The code skips training until at least 64 experiences are saved
            if len(replay_buffer) >= batch_size:
                # Retrieve random mini-batches from memory
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                
                # Predict current action Q-values: Q(s, a)
                # gather() is a method of torch.Tensor (a PyTorch Tensor object).
                #Documentation: https://pytorch.org/docs/stable/generated/torch.Tensor.gather.html
                # gather(1, b_actions) uses b_actions as index markers to select and extract only the Q-value corresponding to the specific action chosen for each sample in the batch.
                current_q = policy_net(b_states).gather(1, b_actions)
                
                # Predict maximum subsequent action values using target network: max Q_target(s', a')
                with torch.no_grad():
                    # next_q : the target network's estimate of the best possible total reward achievable starting from the next state
                    next_q = target_net(b_next_states).max(1)[0].unsqueeze(1) # max(1)[0]: the highest numerical Q-values for the next state.
                    # Compute temporal target value: r + gamma * max Q_target(s', a') * (1 - done)
                    # b_rewards : the immediate numerical reward received right after taking the action.
                    # gamma : The discount factor (e.g., 0.99).
                    # b_dones : a binary flag indicating whether the episode ended (1) or not (0).
                    # If the episode did end (the pole fell over, done = 1), then (1.0 - 1.0) = 0.0
                    target_q = b_rewards + (gamma * next_q * (1.0 - b_dones))

                # Minimize Mean Squared Error (MSE) loss
                # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
                loss = nn.functional.mse_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Synchronize Target Network periodically to stabilize training
            if global_steps % target_sync_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Decay epsilon exploration score
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % 5 == 0:
            print(f"  Episode [{episode+1}/{episodes}] | Score: {episode_reward} | Exploration Epsilon: {epsilon:.4f}")

    env.close()
    print("DQN compilation and training loop validated successfully!")

if __name__ == "__main__":
    main()
