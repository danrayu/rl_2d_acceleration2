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
        self.max_speed = 3.0
        self.max_acceleration = 0.05

        # Agent and target
        self.agent_radius = 10
        self.target_radius = 40
        self.current_step = 0
        self.path_length = 0

        # Initialize agent state (x, y, vx, vy)
        self.state = np.zeros(4)  # [x, y, vx, vy]

        # Action space: acceleration in x and y
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: [x, y, vx, vy, target_x, target_y]
        self.observation_space = spaces.Box(low=np.array([-self.width / 2, -self.height / 2, -self.max_speed, -self.max_speed, -self.width / 2, -self.height / 2]),
                                            high=np.array([self.width / 2, self.height / 2, self.max_speed, self.max_speed, self.width / 2, self.height / 2]),
                                            dtype=np.float32)

        # Target position
        self.target_pos = np.array([random.uniform(-self.width / 2, self.width / 2),
                                    random.uniform(-self.height / 2, self.height / 2)])

        # Roadblock positions (x, y, width, height)
        self.roadblocks = []
        roadblock_margin_x = 50
        roadblock_margin_y = 50
        roadblock_width = 20
        roadblock_height = 500
        roadblock_count = 5
        roadblock_start_pos_x = -self.width/2 + roadblock_margin_x
        roadblock_start_pos_y = -self.height/2 + roadblock_margin_y
        spacing = (self.width - roadblock_count * roadblock_width - roadblock_margin_x * 2) / roadblock_count * 1.5  # Increase the spacing by 1.5x
        for i in range(roadblock_count):
            init_x = roadblock_start_pos_x + i * spacing
            init_y = random.uniform(-self.height/2 + roadblock_margin_y, self.height/2 - roadblock_height - roadblock_margin_y)
            self.roadblocks.append((init_x, init_y, roadblock_width, roadblock_height))

        # Setup for rendering
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            
    def check_collision(self, agent_rect):
        """Check if agent collides with any roadblock."""
        for block in self.roadblocks:
            if agent_rect.colliderect(block):  # Using pygame's built-in collision detection
                return True
        return False

    def reset(self, seed=None, **kwargs):
        self.path_length = 0
        self.current_step = 0
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        # Randomize agent position and reset velocity
        self.state = np.zeros(4)  # [x, y, vx, vy]
        
        # Ensure agent does not spawn on or near a roadblock
        valid_spawn = False
        while not valid_spawn:
            self.state[0] = random.uniform(-self.width / 2, self.width / 2)
            self.state[1] = random.uniform(-self.height / 2, self.height / 2)
            margin = 10
            agent_rect = pygame.Rect(self.state[0] - margin - self.agent_radius, self.state[1] - self.agent_radius - margin, self.agent_radius * 2 + margin, self.agent_radius * 2 + margin)
            
            valid_spawn = not self.check_collision(agent_rect)
        # Randomize target position
        self.target_pos = np.array([random.uniform(-self.width / 2, self.width / 2),
                                    random.uniform(-self.height / 2, self.height / 2)])

        observation = np.concatenate((self.state, self.target_pos))  # [x, y, vx, vy, target_x, target_y]
        if self.render_mode == "human":
            self.render()
        return observation, {}

    def step(self, action):
        ax, ay = action
        ax = np.clip(ax, -self.max_acceleration, self.max_acceleration)
        ay = np.clip(ay, -self.max_acceleration, self.max_acceleration)

        # Update velocity and position
        self.state[2] += ax
        self.state[3] += ay
        speed = np.sqrt(self.state[2]**2 + self.state[3]**2)
        if speed > self.max_speed:
            self.state[2] *= (self.max_speed / speed)
            self.state[3] *= (self.max_speed / speed)

        self.state[0] += self.state[2]
        self.state[1] += self.state[3]
        self.path_length += abs(speed)

        # Compute the distance between agent and target
        dist = np.linalg.norm(self.state[:2] - self.target_pos)

        # Create agent's bounding box (as a pygame.Rect)
        agent_rect = pygame.Rect(self.state[0] - self.agent_radius, self.state[1] - self.agent_radius, self.agent_radius * 2, self.agent_radius * 2)

        # Check if agent collided with any roadblock
        if self.check_collision(agent_rect):
            done = True
            reward = -10  # Penalty for hitting a roadblock
            print("Collision with roadblock!")  # This will print when the agent collides with a roadblock
            return self.get_observation(), reward, done, False, {}

        # Check if agent has collided with the target (distance <= sum of radii)
        done = dist <= self.agent_radius + self.target_radius
        reward = -dist / 10000
        if done:
            reward += 100
            reward += (self.max_speed - speed) * 30
            reward += max(0, self.path_length / 25)

        observation = np.concatenate((self.state, self.target_pos))  # [x, y, vx, vy, target_x, target_y]
        if self.render_mode == "human":
            self.render()
        self.current_step += 1
        if self.current_step >= 1000:
            done = True
            reward += -100
        return observation, reward, done, False, {}



    def render(self):
        if self.render_mode == "human":
            self.screen.fill((255, 255, 255))

            # Draw agent
            pygame.draw.circle(self.screen, (0, 255, 0), (int(self.state[0] + self.width / 2), int(self.state[1] + self.height / 2)), self.agent_radius)

            # Draw target
            pygame.draw.circle(self.screen, (255, 0, 0), (int(self.target_pos[0] + self.width / 2), int(self.target_pos[1] + self.height / 2)), self.target_radius)

            # Draw roadblocks
            for block in self.roadblocks:
                block_x, block_y, block_w, block_h = block
                pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(block_x + self.width / 2, 
                                                                        block_y + self.height / 2, 
                                                                        block_w, block_h))

            pygame.display.flip()
            self.clock.tick(60)  # Control frame rate

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
