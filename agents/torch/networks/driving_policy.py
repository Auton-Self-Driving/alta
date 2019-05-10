import torch
import torch.nn as nn
from torch.distributions import Normal

from .abstract_policy import DeterministicPolicy
from .block import FcBlock


class DrivingDeterministicPolicy(DeterministicPolicy):
    """ 
    Inputs:
        - obs["image_features"]: tensor of size (batch_size, 512)
        - obs["speed"]: tensor of size (batch_size, 1)
        - obs["branch_mask"]: tensor of size (batch_size, 4)
    Output:
        - action: tensor of size (batch_size, 2) in range [-1, 1]
    """
    def __init__(self, action_dim=2):
        super(DrivingDeterministicPolicy, self).__init__()
        
        # Fully connected layers for speed input
        # self.fc_speed = nn.Sequential(
        #     FcBlock(1, 128),
        #     FcBlock(128, 128)
        # )
        
        # Fully connected layers to merge image and speed features
        # self.fc_merge = FcBlock(512 + 128, 512)
        
        # Fully connected output branches 
        """
        self.fc_branches = nn.ModuleList(
            [nn.Sequential(
                FcBlock(512, 256),
                FcBlock(256, 256),
                nn.Linear(256, 2),
                nn.Tanh()
            ) for _ in range(4)]
        )
        """
        self.fc = nn.Sequential(
            FcBlock(512, 256),
            FcBlock(256, 256),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):        
        # Process speed
        # x = self.fc_speed(obs["speed"])
        
        # Merge speed and image features
        # x = torch.cat((obs["image_features"], x), dim=1)
        # x = self.fc_merge(x)
        
        x = obs["image_features"]

        """
        # Compute output branches
        x = [branch(x) for branch in self.fc_branches]
        x = torch.stack(x)

        # Select output branch with branch mask
        branch_mask = obs["branch_mask"].t().unsqueeze(2)
        x = torch.sum(x * branch_mask, dim=0)
        """
        x = self.fc(x)
        
        return x