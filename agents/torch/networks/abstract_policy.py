from typing import Dict, Union

import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution


class DeterministicPolicy(nn.Module):
    def __init__(self):
        super(DeterministicPolicy, self).__init__()
        
    def forward(self, 
                obs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        pass
    
    
class StochasticPolicy(nn.Module):
    def __init__(self):
        super(StochasticPolicy, self).__init__()
        
    def forward(self, 
                obs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Distribution:
        pass