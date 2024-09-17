import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from model import Model
from game import Game, Action

class A2CAgent:
    def __init__(self, model):
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def initialize_model(self, env):
        self.model.action_value(env.cur_obs())
        self.model.top_action(env.cur_obs())

    def generate_experience_batch(self, env, batch_size):
        action_ids = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size, 2, Game.HEIGHT+2, Game.WIDTH+2))

        ep_rews = [0.0]
        next_obs = env.cur_obs()
        for step in range(batch_size):
            observations[step] = next_obs.copy()
            action_ids[step], values[step] = self.model.action_value(next_obs)
            next_obs, rewards[step], dones[step] = env.step(self._action_from_id(action_ids[step]))

            ep_rews[-1] += rewards[step]
            if dones[step]:
                ep_rews.append(0.0)
                next_obs = env.reset()

        _, next_value = self.model.action_value(next_obs)
        returns, advs = self._returns_advantages(rewards, dones, values, next_value)
        acts_and_advs = np.concatenate([action_ids[:, None], advs[:, None]], axis=-1)
        return observations, acts_and_advs, returns

    def select_top_action(self, obs):
        action_id = self.model.top_action(obs)
        return self._action_from_id(action_id)

    def save_model(self):
        torch.save(self.model.state_dict(), 'saved_model/weights.pth')

    def load_model_if_previously_saved(self):
        if os.path.exists('saved_model/weights.pth'):
            self.model.load_state_dict(torch.load('saved_model/weights.pth'))

    def load_pretrained_model(self):
        if os.path.exists('pretrained_model/weights.pth'):
            self.model.load_state_dict(torch.load('pretrained_model/weights.pth'))

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        return self.params['value'] * F.mse_loss(value, returns)

    def _logits_loss(self, acts_and_advs, logits):
        actions, advantages = torch.split(acts_and_advs, 1, dim=-1)
        actions = actions.long()
        weighted_sparse_ce = F.cross_entropy(logits, actions.squeeze(-1), reduction="none")
        policy_loss = (weighted_sparse_ce * advantages).mean()
        entropy_loss = -self.params['entropy'] * torch.mean(torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1))
        return policy_loss + entropy_loss

    def _action_from_id(self, action_id):
        return self.model.all_possible_actions_in_game[action_id]

    def train(self, observations, acts_and_advs, returns):
        observations = torch.FloatTensor(observations)
        acts_and_advs = torch.FloatTensor(acts_and_advs)
        returns = torch.FloatTensor(returns)

        logits, values = self.model(observations)

        value_loss = self._value_loss(returns, values.squeeze(-1))
        logits_loss = self._logits_loss(acts_and_advs, logits)
        loss = value_loss + logits_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
