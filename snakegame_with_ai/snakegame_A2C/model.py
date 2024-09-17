import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import Direction

class EntryLayer(nn.Module):
    def __init__(self):
        super(EntryLayer, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class RepeatedLayer(nn.Module):
    def __init__(self):
        super(RepeatedLayer, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class PolicyHeadLayer(nn.Module):
    def __init__(self, num_of_actions):
        super(PolicyHeadLayer, self).__init__()
        self.conv = nn.Conv2d(64, 2, kernel_size=1)
        self.fc = nn.Linear(2 * 22 * 22, num_of_actions)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ValueHeadLayer(nn.Module):
    def __init__(self):
        super(ValueHeadLayer, self).__init__()
        self.conv = nn.Conv2d(64, 1, kernel_size=1)
        self.fc1 = nn.Linear(22 * 22, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.all_possible_actions_in_game = [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN]

        self.entry = EntryLayer()
        self.repeated = nn.Sequential(*[RepeatedLayer() for _ in range(3)])
        self.actor_head = PolicyHeadLayer(len(self.all_possible_actions_in_game))
        self.critic_head = ValueHeadLayer()

    def forward(self, x):
        x = self.entry(x)
        x = self.repeated(x)
        actor_output = self.actor_head(x)
        critic_output = self.critic_head(x)
        return actor_output, critic_output

    def action_value(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
        logits, value = self.forward(obs)
        action = torch.distributions.Categorical(logits=logits).sample()
        return action.item(), value.item()

    def top_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
        logits, _ = self.forward(obs)
        return torch.argmax(logits, dim=1).item()

    def get_variables(self):
        return self.state_dict()

    def set_variables(self, variables):
        self.load_state_dict(variables)
