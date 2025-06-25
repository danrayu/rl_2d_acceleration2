import gymnasium as gym
from stable_baselines3 import SAC
from env import AccelerationNavigationEnv
import torch
import time
import os 

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create the environment
env = AccelerationNavigationEnv(render_mode="human")
obs, info = env.reset()  # Reset environment

# Load the trained model
model = SAC.load("model/1750830361/55.zip", env)

# Reset environment and start testing
done = False
total_reward = 0

for ep in range(0, 10):
    obs, info = env.reset()  # Reset environment
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _  = env.step(action)
        total_reward += reward

# Print the final accumulated reward
print(f"Final Total Reward: {total_reward}")
