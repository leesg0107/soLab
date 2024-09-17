import numpy as np
from tag_game_env import TagEnv
import pygame

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, state_size, state_size, state_size, action_size))

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[tuple(state)]) % self.action_size

    def train(self, state, action, reward, next_state, done):
        target = reward + self.discount_factor * np.max(self.q_table[tuple(next_state)]) * (not done)
        self.q_table[tuple(state)][action] = (1 - self.learning_rate) * self.q_table[tuple(state)][action] + self.learning_rate * target

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def state_to_index(state):
    return tuple(state.astype(int))

# Initialize environment and agent
env = TagEnv()
agent = QLearningAgent(env.GRID_SIZE, env.GRID_SIZE, env.GRID_SIZE, env.GRID_SIZE, env.action_space.n)

# Training parameters
n_episodes = 10000
max_steps = 100

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.get_action(state_to_index(state))
        next_state, reward, done, _ = env.step(action)
        
        agent.train(state_to_index(state), action, reward, state_to_index(next_state), done)
        
        state = next_state
        total_reward += reward

        if done:
            break

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

print("Training completed!")

# Test the trained agent
state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    action = agent.get_action(state_to_index(state))
    state, reward, done, _ = env.step(action)
    total_reward += reward

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

print(f"Test episode finished. Total Reward: {total_reward}")
env.close()
