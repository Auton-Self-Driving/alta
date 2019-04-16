from typing import Dict, Union

import torch
import torch.nn as nn


class StateValue(nn.Module):
    def __init__(self):
        super(StateValue, self).__init__()

    def forward(self, 
                obs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        pass