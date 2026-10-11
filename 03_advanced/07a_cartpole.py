"""
CARTPOLE-v1 RL ENVIRONMENT AND TRAINING PARAMETERS:

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

import gymnasium as gym

# 1. Create the environment (headless mode - no visual window)
env = gym.make("CartPole-v1")

# 2. Reset the environment to start a new episode
observation, info = env.reset()

print(f"Initial Observation: {observation}")

# 3. Run a simple loop for 100 timesteps
for step in range(100):
    # Select a random action from the action space (0 = push left, 1 = push right)
    action = env.action_space.sample()

    # Apply the action to the environment
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step + 1} | Action: {action} | Reward: {reward}")

    # If the pole falls over (terminated) or hits the time limit (truncated), reset    
    # The episode terminates when the pole becomes unstable:
    # 1.Pole angle exceeds ±12 degrees from vertical
    # 2.Cart position exceeds ±2.4 units from center

    if terminated or truncated:
        print("Episode finished. Resetting environment..." + str(observation))
        if terminated:
            print("Pole fell over.")
        elif truncated:
            print("Time limit reached.")
        observation, info = env.reset()

# 4. Clean up and close the window
env.close()