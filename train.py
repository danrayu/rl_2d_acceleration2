import gymnasium as gym
from stable_baselines3 import DDPG
from env import AccelerationNavigationEnv
import torch

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create the environment
env = AccelerationNavigationEnv(render_mode="humann")

# Create the DDPG model
model = DDPG('MlpPolicy', env, verbose=1, device=device)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("ddpg_acceleration_navigation_model")

# Use the trained model to make predictions (e.g., move the agent)
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
