import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pass