import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from sklearn.neighbors import KDTree
from pyflann import FLANN

from .utils import mlp, weight_init


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q = self.Q(obs_action)

        self.outputs['q'] = q

        return q


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, logger, step):
        for k, v in self.outputs.items():
            logger.log_histogram(f'train_critic/{k}_hist', v, step)

        assert len(self.Q1) == len(self.Q2)
        for i, (m1, m2) in enumerate(zip(self.Q1, self.Q2)):
            assert type(m1) == type(m2)
            if type(m1) is nn.Linear:
                logger.log_param(f'train_critic/q1_fc{i}', m1, step)
                logger.log_param(f'train_critic/q2_fc{i}', m2, step)


class MemoryCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, memory_size=5e5):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_size = memory_size

        self.memory = {
            'obs': torch.zeros((0, obs_dim)).cpu(),
            'action': torch.zeros((0, action_dim)).cpu(),
            'reward': torch.zeros((0, 1)).cpu(),
            'Q': torch.zeros((0, 1)).cpu()
        }

        self.kdtree = FLANN()

    def update(self, obs, action, reward, q_target):
        self.memory['obs'] = torch.cat([self.memory['obs'], obs.cpu()], dim=0)
        self.memory['action'] = torch.cat([self.memory['action'], action.cpu()], dim=0)
        self.memory['reward'] = torch.cat([self.memory['reward'], reward.cpu()], dim=0)
        self.memory['Q'] = torch.cat([self.memory['Q'], q_target.cpu()], dim=0)

        if len(self.memory['obs']) > self.memory_size:
            self.memory['obs'] = self.memory['obs'][-self.memory_size:]
            self.memory['action'] = self.memory['action'][-self.memory_size:]
            self.memory['reward'] = self.memory['reward'][-self.memory_size:]
            self.memory['Q'] = self.memory['Q'][-self.memory_size:]

        state_action = torch.cat([self.memory['obs'], self.memory['action']], dim=1)
        self.kdtree.build_index(state_action.cpu().detach().numpy())

    def forward(self, obs, action):
        if len(self.memory['Q']) < 50:
            return torch.zeros(obs.size(0),1).cuda()

        state_action = torch.cat([obs, action], dim=1)
        neighbors, dists = self.kdtree.nn_index(state_action.cpu().detach().numpy(), num_neighbors=10)
        Qs = self.memory['Q'][neighbors].squeeze(-1)
        weights = F.softmax(torch.tensor(dists), dim=1)
        return (weights * Qs).sum(dim=1).cuda()