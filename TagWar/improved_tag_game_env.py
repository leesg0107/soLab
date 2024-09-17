import pygame
import numpy as np
import gym
from gym import spaces

class ImprovedTagEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ImprovedTagEnv, self).__init__()

        # Constants
        self.WIDTH, self.HEIGHT = 600, 600
        self.GRID_SIZE = 20
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        # Action and observation space
        self.action_space = spaces.Discrete(9)  # 8 directions + stay
        self.observation_space = spaces.Box(low=0, high=self.GRID_SIZE-1, 
                                            shape=(4,), dtype=np.float32)

        # Initialize Pygame
        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()

        # Initialize agents
        self.reset()

    def reset(self):
        self.tagger = np.random.randint(0, self.GRID_SIZE, size=2)
        self.runner = np.random.randint(0, self.GRID_SIZE, size=2)
        
        # Ensure tagger and runner start at different positions
        while np.array_equal(self.tagger, self.runner):
            self.runner = np.random.randint(0, self.GRID_SIZE, size=2)
        
        return self._get_obs()

    def step(self, action):
        # Move tagger based on action
        self._move_agent(self.tagger, action)
        
        # Move runner using simple heuristic
        self._move_runner()

        # Check if game is done
        done = np.array_equal(self.tagger, self.runner)
        
        # Compute reward
        reward = self._compute_reward(done)

        return self._get_obs(), reward, done, {}

    def _move_agent(self, agent, action):
        # Action mapping: 0=stay, 1=up, 2=up-right, 3=right, 4=down-right, 
        #                 5=down, 6=down-left, 7=left, 8=up-left
        dx, dy = [(0,0), (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)][action]
        agent[0] = (agent[0] + dx) % self.GRID_SIZE
        agent[1] = (agent[1] + dy) % self.GRID_SIZE

    def _move_runner(self):
        # Simple heuristic: move away from tagger
        dx = self.runner[0] - self.tagger[0]
        dy = self.runner[1] - self.tagger[1]
        
        if abs(dx) > abs(dy):
            move = np.sign(dx)
            self.runner[0] = (self.runner[0] + move) % self.GRID_SIZE
        else:
            move = np.sign(dy)
            self.runner[1] = (self.runner[1] + move) % self.GRID_SIZE

    def _compute_reward(self, done):
        if done:
            return 10  # High reward for catching
        else:
            distance = np.linalg.norm(self.tagger - self.runner)
            return 1 / (distance + 1)  # Reward inversely proportional to distance

    def _get_obs(self):
        return np.concatenate([self.tagger, self.runner]).astype(np.float32)

    def render(self, mode='human'):
        if self.screen is None and mode == 'human':
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Improved Tag Game")

        if mode == 'human':
            self.screen.fill(self.WHITE)

            # Draw grid
            for x in range(0, self.WIDTH, self.CELL_SIZE):
                pygame.draw.line(self.screen, self.BLACK, (x, 0), (x, self.HEIGHT))
            for y in range(0, self.HEIGHT, self.CELL_SIZE):
                pygame.draw.line(self.screen, self.BLACK, (0, y), (self.WIDTH, y))

            # Draw agents
            pygame.draw.rect(self.screen, self.RED, 
                             (self.tagger[0]*self.CELL_SIZE, self.tagger[1]*self.CELL_SIZE, 
                              self.CELL_SIZE, self.CELL_SIZE))
            pygame.draw.rect(self.screen, self.BLUE, 
                             (self.runner[0]*self.CELL_SIZE, self.runner[1]*self.CELL_SIZE, 
                              self.CELL_SIZE, self.CELL_SIZE))

            pygame.display.flip()
            self.clock.tick(10)

        elif mode == 'rgb_array':
            return np.transpose(
                pygame.surfarray.array3d(self.screen), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None