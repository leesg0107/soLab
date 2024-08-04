import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np

n_train_processes = 3
learing_rate=0.0002
update_interval=5
gamma=0.98
max_train_steps=60000
PRINT_INTERVAL=update_interval*100

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic,self).__init__()
        self.fc1=nn.Linear(4,256)
        self.fc_pi=nn.Linear(256,2)
        self.fc_v=nn.Linear(256,1)
    def pi(self,x,softmax_dim=1):
        x=F.relu(self.fc1(x))
        x=self.fc_pi(x)
        prob=F.softmax(x,dim=softmax_dim)
        return prob

    def v(self,x):
        x=F.relu(self.fc1(x))
        v=self.fc_v(x)
        return v
