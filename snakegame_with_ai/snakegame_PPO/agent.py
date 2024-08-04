import numpy as np
import torch
import torch.optim as optim
from model import PPOModel

class PPOAgent:
    def __init__(self, input_size, output_size, lr, gamma, lmbda, epochs, eps_clip):
        self.model = PPOModel(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, _ = self.model(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, states, actions, rewards, dones, next_states):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float)

        old_probs, _ = self.model(states)
        old_probs = old_probs.gather(1, actions)

        for _ in range(self.epochs):
            probs, values = self.model(states)
            probs = probs.gather(1, actions)
            ratio = probs / old_probs.detach()
            advantages = rewards + self.gamma * self.model(next_states)[1] * (1 - dones) - values
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * torch.square(values - rewards)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
