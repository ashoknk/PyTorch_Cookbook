"""
CARTPOLE-v1 RL ENVIRONMENT AND TRAINING PARAMETERS:

Gymnasium is an open-source Python library that provides a standardized API 
for developing, testing, and benchmarking single-agent reinforcement learning (RL) algorithms.
Gymnasium standardizes how an AI agent interacts with a virtual environment by providing 
a rigid structure for four essential concepts:

  - Observation State Space Size (4):
    The AI receives 4 numerical values describing the physical state at each step:
    [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity].

  - Action Space Size (2):
    The discrete choices available to the agent at each timestep:
    - 0: Push cart left
    - 1: Push cart right

  - Score (Episode Reward):
    The total number of consecutive timesteps the pole remained balanced upright 
    before falling or hitting terminal limits. Higher values indicate better performance.

  - Exploration Epsilon:
    The threshold probability governing the Epsilon-Greedy strategy (Exploration vs. Exploitation).
    As it decays over time (e.g., from 1.0 down to lower values), the agent shifts from 
    taking random exploratory actions to relying on its neural network's learned policy.
"""

import torch
import gymnasium as gym

# 1. Create the environment (headless mode - no visual window)
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"Observation State Space size: {state_dim}, Action Space size: {action_dim}")
    
# 2. Reset the environment to start a new episode
observation, _ = env.reset()
cart_pos, cart_vel, pole_angle, pole_vel = observation
pole_angle_degrees = pole_angle * 180 / 3.14159  # Convert radians to degrees
print(f"Initial Observation: - Cart Pos: {cart_pos:.3f}m, Cart Vel: {cart_vel:.3f}m/s")
print(f"             Pole Angle: {pole_angle:.3f}rad ({pole_angle_degrees:.1f}°), Pole Vel: {pole_vel:.3f}rad/s")
        

print("\n")
# 3. Create action label dictionary for readable output
action_labels = {0: "push left", 1: "push right"}
episode_reward = 0

# 4. Run a simple loop for 100 timesteps
for step in range(100):
    # Select a random action from the action space (0 = push left, 1 = push right)
    action = env.action_space.sample()

    # Apply the action to the environment
    observation, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward

    if(step % 10 == 0):
      action_label = action_labels[action]
      print(f"Step {step + 1} | Action: {action_label} | Reward: {episode_reward}")

    # If the pole falls over (terminated) or hits the time limit (truncated), reset
    # The episode terminates when the pole becomes unstable:
    # 1.Pole angle exceeds ±12 degrees from vertical
    # 2.Cart position exceeds ±2.4 units from center

    if terminated or truncated:
        action_label = action_labels[action]
        # Show both the actual reward and what a penalty-based system might show
        penalty_reward = -1.0 if terminated else 0.0  # Custom penalty for falling
        print(f"FINAL Step {step + 1} | Action: {action_label} | Actual Reward: {reward} | Custom Penalty: {penalty_reward}")

        # Decode the observation values for better understanding
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        pole_angle_degrees = pole_angle * 180 / 3.14159  # Convert radians to degrees

        print(f"\tFinal State - Cart Pos: {cart_pos:.3f}m, Cart Vel: {cart_vel:.3f}m/s")
        print(f"             Pole Angle: {pole_angle:.3f}rad ({pole_angle_degrees:.1f}°), Pole Vel: {pole_vel:.3f}rad/s")
        print(f"\tRaw observation: {observation}")
        

        if terminated:
            if abs(pole_angle_degrees) > 12:
                print(f"\tPole fell over - Angle {pole_angle_degrees:.1f}° exceeded ±12° limit!")
            elif abs(cart_pos) > 2.4:
                print(f"\tCart went too far - Position {cart_pos:.3f}m exceeded ±2.4m limit!")
        elif truncated:
            print("\tTime limit reached.")
        observation, _ = env.reset()
        print("\n")


# 4. Clean up and close the window
env.close()


# ================gather======
"""
torch.Tensor.gather() is an advanced multi-index selection tool used to collect or slice out 
specific values from a source tensor along a specified dimension using an index tensor.
Documentation: https://pytorch.org/docs/stable/generated/torch.Tensor.gather.html
"""

# Create a 2x3 source tensor
source = torch.tensor([
    [10, 20, 30],
    [40, 50, 60]])

# =====1. Gathering along columns (dim=1) =====
# We want to pick elements from each row by their column indices.
# Row 0: pick index 0 (10) and index 2 (30)
# Row 1: pick index 1 (50) and index 1 (50)
index_dim1 = torch.tensor([
    [0, 2],
    [1, 1]])

out1 = source.gather(dim=1, index=index_dim1)
print(out1)
# Output: tensor([[10, 30],
#                [50, 50]])

# ===== 2. Gathering along rows (dim=0) =====
# We want to pick elements from each column by their row indices.
# Column 0: pick row 0 (10)
# Column 1: pick row 1 (50)
# Column 2: pick row 0 (30)
index_dim0 = torch.tensor([[0, 1, 0]])

out0 = source.gather(dim=0, index=index_dim0)
print(out0)
# Output: tensor([[10, 50, 30]])
