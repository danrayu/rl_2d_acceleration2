import gymnasium as gym
from stable_baselines3 import SAC
from env import AccelerationNavigationEnv
import torch
import time
import os 

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


timestamp = round(time.time())
logdir = "logs"
modeldir = f"model/{timestamp}"
os.makedirs(logdir, exist_ok=True)
os.makedirs(modeldir, exist_ok=True)

# Create the environment
env = AccelerationNavigationEnv(render_mode="humaan")

# Create the DDPG model
model = SAC('MlpPolicy', env, verbose=1, device=device, tensorboard_log=logdir)
model.load("model/1750789134/21.zip")

# Train the agent
for i in range(1, 1000):
    model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=f"run_{timestamp}")
    model.save(f"{modeldir}/{i}")
    

# Save the model
model.save("ddpg_acceleration_navigation_model")

# Use the trained model to make predictions (e.g., move the agent)
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
