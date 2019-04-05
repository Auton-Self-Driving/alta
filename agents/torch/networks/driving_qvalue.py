import torch
import torch.nn as nn

from .block import FcBlock
from .abstract_qvalue import QValue


class DrivingQValue(QValue):
    """ 
    Inputs:
        - obs["image_features"]: tensor of size (batch_size, 512)
        - obs["speed"]: tensor of size (batch_size, 1)
        - obs["branch_mask"]: tensor of size (batch_size, 4)
        - action: tensor of size (batch_size, 2)
    Output:
        - tensor of size (batch_size, 1)
    """
    def __init__(self):
        super(DrivingQValue, self).__init__()
        
        # Fully connected layers for speed input
        self.fc_speed = nn.Sequential(
            FcBlock(1, 128),
            FcBlock(128, 128)
        )
        
        # Fully connected layers for action input
        self.fc_action = nn.Sequential(
            FcBlock(2, 128),
            FcBlock(128, 128)
        )
        
        # Fully connected layers to merge image, speed and action features
        self.fc_merge = FcBlock(512 + 128 + 128, 512)
        
        # Fully connected output branches 
        """
        self.fc_branches = nn.ModuleList(
            [nn.Sequential(
                FcBlock(512, 256),
                FcBlock(256, 256),
                nn.Linear(256, 1)
            ) for _ in range(4)]
        )
        """
        self.fc = nn.Sequential(
            FcBlock(512, 256),
            FcBlock(256, 256),
            nn.Linear(256, 2),
            nn.Tanh()
        )

    def forward(self, obs, action):
        # Process speed and action
        x1 = self.fc_speed(obs["speed"])
        x2 = self.fc_action(action)
        
        # Merge image, speed and action features
        x = torch.cat((obs["image_features"], x1, x2), dim=1)
        x = self.fc_merge(x)
        
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