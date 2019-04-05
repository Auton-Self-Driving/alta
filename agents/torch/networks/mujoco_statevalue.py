import torch
import torch.nn as nn

from .abstract_statevalue import StateValue


class MujocoStateValue(StateValue):
    def __init__(self, state_dim):
        super(MujocoStateValue, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, x):
        return self.net(x)