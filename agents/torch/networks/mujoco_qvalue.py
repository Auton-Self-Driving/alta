import torch
import torch.nn as nn

from .abstract_qvalue import QValue


class MujocoQValue(QValue):
    def __init__(self, state_dim, action_dim, pixel_obs=False):
        super(MujocoQValue, self).__init__()
        self.pixel_obs = pixel_obs
        
        if pixel_obs:
            self.fc_action = nn.Sequential(
                nn.Linear(action_dim, 100),
                nn.ReLU(),
            )
            action_dim = 100

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, obs, action):
        if self.pixel_obs:
            x = self.fc_action(action)
            x = self.net(torch.cat([obs["image_features"], x], 1))
        else:
            x = self.net(torch.cat([obs, action], 1))
        return x