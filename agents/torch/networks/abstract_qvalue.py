from typing import Dict, Union

import torch
import torch.nn as nn


class QValue(nn.Module):
    def __init__(self):
        super(QValue, self).__init__()

    def forward(self, 
                obs: Union[Dict[str, torch.Tensor], torch.Tensor], 
                action: torch.Tensor) -> torch.Tensor:
        pass