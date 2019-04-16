import torch
import torch.nn as nn

from .abstract_policy import DeterministicPolicy, StochasticPolicy
from .distribution import TanhNormal


class MujocoDeterministicPolicy(DeterministicPolicy):
    def __init__(self, state_dim, action_dim, pixel_obs=False):
        super(MujocoDeterministicPolicy, self).__init__()
        self.pixel_obs = pixel_obs
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        if self.pixel_obs:
            action = self.net(obs["image_features"])
        else:
            action = self.net(obs)
        return action
    
    
class MujocoTanhGaussianPolicy(StochasticPolicy):
    def __init__(self, state_dim, action_dim):
        super(MujocoTanhGaussianPolicy, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(300, action_dim)
        self.fc_logstd = nn.Linear(300, action_dim)

    def forward(self, obs):
        x = self.net(obs)
        normal_mean = self.fc_mean(x)
        normal_std = torch.exp(self.fc_logstd(x))            
        distrib = TanhNormal(normal_mean, normal_std)
        return distrib