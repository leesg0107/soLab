import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from improved_tag_game_env import ImprovedTagEnv
import pygame

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                target = (reward + self.gamma *
                          np.amax(self.model(next_state).cpu().data.numpy()))
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_f = self.model(state)
            target_f[0][action] = target
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize environment and agent
env = ImprovedTagEnv()
agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

# Training parameters
n_episodes = 1000
max_steps = 200
batch_size = 32

# Training loop
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward

        if done:
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

print("Training completed!")

# Test the trained agent
state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    action = agent.act(state)
    state, reward, done, _ = env.step(action)
    total_reward += reward

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

print(f"Test episode finished. Total Reward: {total_reward}")
env.close()