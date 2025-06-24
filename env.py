import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# Create the environment
class AccelerationNavigationEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super(AccelerationNavigationEnv, self).__init__()

        # Parameters for the environment
        self.width = 800
        self.height = 600

        self.max_speed = 2.0
        self.max_acceleration = 0.1

        # Agent and target
        self.agent_radius = 10
        self.target_radius = 40
        self.current_step = 0

        # Initialize agent state (x, y, vx, vy)
        self.state = np.zeros(4)  # [x, y, vx, vy]

        # Action space: acceleration in x and y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [x, y, vx, vy, target_x, target_y]
        # Agent's position (x, y), velocity (vx, vy) + target's position (target_x, target_y)
        self.observation_space = spaces.Box(low=np.array([-self.width / 2, -self.height / 2, -self.max_speed, -self.max_speed, -self.width / 2, -self.height / 2]),
                                            high=np.array([self.width / 2, self.height / 2, self.max_speed, self.max_speed, self.width / 2, self.height / 2]),
                                            dtype=np.float32)

        # Target position
        self.target_pos = np.array([random.uniform(-self.width / 2, self.width / 2),
                                    random.uniform(-self.height / 2, self.height / 2)])

        # Setup for rendering
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, **kwargs):
        # Optional: Set the seed for random number generation (if needed)
        self.current_step = 0
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Randomize agent position and reset velocity
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.state[0] = random.uniform(-self.width / 2, self.width / 2)
        self.state[1] = random.uniform(-self.height / 2, self.height / 2)

        # Randomize target position
        self.target_pos = np.array([random.uniform(-self.width / 2, self.width / 2),
                                    random.uniform(-self.height / 2, self.height / 2)])

        # Observation now includes the target's position
        observation = np.concatenate((self.state, self.target_pos))  # [x, y, vx, vy, target_x, target_y]

        if self.render_mode == "human":
            self.render()

        return observation, {}

    def step(self, action):
        # Normalize the action to control acceleration in both axes
        ax, ay = action
        ax = np.clip(ax, -self.max_acceleration, self.max_acceleration)
        ay = np.clip(ay, -self.max_acceleration, self.max_acceleration)

        # Update velocity based on acceleration
        self.state[2] += ax
        self.state[3] += ay

        # Apply maximum speed limit
        speed = np.sqrt(self.state[2]**2 + self.state[3]**2)
        if speed > self.max_speed:
            self.state[2] *= (self.max_speed / speed)
            self.state[3] *= (self.max_speed / speed)

        # Update position based on velocity
        self.state[0] += self.state[2]
        self.state[1] += self.state[3]

        # Compute the distance between agent and target
        dist = np.linalg.norm(self.state[:2] - self.target_pos)

        # Check if agent has collided with the target (distance <= sum of radii)
        done = dist <= self.agent_radius + self.target_radius
        reward = -dist  # Reward is negative distance (closer to target = higher reward)

        # Observation now includes the target's position
        observation = np.concatenate((self.state, self.target_pos))  # [x, y, vx, vy, target_x, target_y]

        if self.render_mode == "human":
            self.render()
        self.current_step += 1
        if self.current_step >= 1000:
            done = True
            reward += -100
        return observation, reward, done, False, {}

    def render(self):
        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Draw agent (circle)
        pygame.draw.circle(self.screen, (0, 255, 0), (int(self.state[0] + self.width / 2), int(self.state[1] + self.height / 2)), self.agent_radius)

        # Draw target (circle)
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.target_pos[0] + self.width / 2), int(self.target_pos[1] + self.height / 2)), self.target_radius)

        pygame.display.flip()
        self.clock.tick(60)  # Control the frame rate

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
