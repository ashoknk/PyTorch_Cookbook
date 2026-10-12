"""
20. Deep Q-Network (DQN) Reinforcement Learning in PyTorch

The `QNetwork` is a neural network that takes the current game state as input and outputs estimated future 
reward scores (Q-values) for every possible move. It acts as the agent's brain, allowing it to choose 
the best action by picking the one with the highest predicted score. 
During training, its predictions are adjusted using MSE loss so its guesses become increasingly accurate.

The main job of QNetwork is to calculate Q-values, and it gives those values to the player 
(the action-selection decision loop) and the loss function (for training).

What is a Q-Value?
    a Q-value is a score that estimates the total future reward the agent will 
    get if it takes a specific action right now and keeps playing smartly afterward.

The main purpose of this code is to train a Deep Q-Network (DQN) AI agent to play 
the CartPole game without human intervention.Instead of writing hardcoded rules 
(e.g., "if pole leans right, move right"), the code gives the neural network complete control. 
The network starts by making random guesses (exploring), records its experiences into memory, 
and gradually learns which moves keep the pole balanced to achieve the highest score.   

1. Reinforcement Learning vs. Supervised Learning:
   - Supervised Learning: Targets are fixed ground-truth labels (e.g., Image Net class IDs).
   - Reinforcement Learning: Targets don't exist upfront! The agent must construct 
     its own targets on the fly using bootstrapping (Reward + Discounted Future Guess).

2. Why target_q Is Dynamic:
   - target_q = b_rewards + gamma * next_q * (1 - b_dones).
   - Because next_q comes from processing b_next_states through target_net, 
     the target must be calculated inside the training loop for every batch.

Learning Objectives:
1. Define a Q-Network mapping environment states to action values.
2. Build an Experience Replay Buffer containing transition trajectories.
3. Apply epsilon-greedy strategies to balance environment exploration and exploitation.
4. Set up stable Q-learning target computations using a target network.
"""

import random
from collections import deque
import numpy as np

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
"""
The `QNetwork` is a neural network that takes the current game state 
as input and outputs estimated future reward scores (Q-values) for every possible move. 

What is a Q-Value?
    a Q-value is a score that estimates the total future reward the agent will 
    get if it takes a specific action right now and keeps playing smartly afterward.
    the agent will earn if it takes action 'a' from state 's' and plays optimally thereafter.

   Q-values are expected point totals (e.g., Left=2.5, Right=0.1).
   Exploitation: Picks the action with the single highest Q-value (e.g., chooses Left because 2.5 > 0.1).
"""
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
        # ex. tensor([[2.5, 0.1]]) --> [Q-value for Action 0 (Left), Q-value for Action 1 (Right)]
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
        # NOTE: without numpy you might see a "TypeError: expected np.ndarray (got list)" error when creating torch tensors
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
    # print(f"Observation State Space : {env.observation_space}, Action Space : {env.action_space}")

    """ Initialize Active and Target networks.
    The target network provides static reference coordinates for temporal differences (TD) loss.
    policy_net (Active Network): 
      This is the live network being trained on every batch. Its weights are updated at every optimization step.   
    target_net (Target Network): 
      This acts as a frozen anchor. It stays completely still while policy_net learns, giving the agent a stable benchmark
    """
    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    # Align target weights initially. Copies all weights and biases from policy_net directly into target_net.
    # As policy_net learns, its weights change at every step, but target_net stays frozen.
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
            # (Exploration) The agent picks a completely random move. 
            if random.random() < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else: 
                # (Exploitation) uses the Neural Network's best calculated guess. 
                # The AI will consult its trained policy_net neural network to choose the smartest move.
                # Run state_t through policy_net() to calculate Q-values and find the index of the highest Q-value (torch.argmax)
                # Calculate the model's smartest decision based on its current trained weights.
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = torch.argmax(policy_net(state_t), dim=1).item() #Left (0) or Right (1)

            # Execute step in Gymnasium
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition in Replay Memory
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

            # 3.2 Optimization Step
            # Checks if there are enough saved game steps inside the ReplayBuffer to fill a mini-batch 
            # (e.g., 64 past steps). For the first few steps of Ep 1, the buffer is empty.
            # The code skips training until at least 64 experiences are saved
            if len(replay_buffer) >= batch_size:
                # Retrieve random mini-batches from memory
                b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample(batch_size)
                
                # Predict current action Q-values: Q(s, a)
                """
                torch.Tensor.gather() is an advanced multi-index selection tool used to slice out .
                gather(1, b_actions) uses b_actions as index markers to select and 
                extract only the Q-value corresponding to the specific action chosen for each sample in the batch.

                b_actions as a filter/index marker to pluck out only the Q-value of the executed action:
                """
                current_q = policy_net(b_states).gather(1, b_actions) #The output is a floating-point score ex. 2.5 or 0.1
                
                """ 
                Predict maximum subsequent action values using target network: max Q_target(s', a')
                next_q : In reinforcement learning, a good action isn't just about immediate reward—it's also 
                about setting yourself up for success in the future. next_q captures that future potential.
                
                target_q  : This is the mathematical "ideal score" for the step you just took. (The Goal)
                            target_q  = Immediate Reward (b_rewards)+ Discounted Future Reward (next_q from target_net).
                b_rewards : the immediate numerical reward received right after taking the action.
                gamma     : The discount factor (e.g., 0.99).
                b_dones   : a binary flag indicating whether the episode ended (1) or not (0).
                            If the episode did end (the pole fell over, done = 1), then (1.0 - 1.0) = 0.0
                """
                with torch.no_grad():
                    # next_q : the target network's estimate of the best possible total reward 
                    # achievable starting from the next state (calculated by the frozen target_net)
                    # target_net predicts Q-values for the next states. .max(1)[0] extracts the maximum future Q-value for the next state.
                    next_q = target_net(b_next_states).max(1)[0].unsqueeze(1) 
                    # Compute temporal target value: r + gamma * max Q_target(s', a') * (1 - done)
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
