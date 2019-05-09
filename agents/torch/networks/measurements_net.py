import torch
import torch.nn as nn

from .block import ConvBlock, FcBlock

class MeasurementNet(nn.Module):
    """
    Input: 
        - image: tensor of size (batch_size, 1)
    Output:
        - image_features: tensor of size (batch_size, 512)
    """

    def __init__(self, dropout_rate=0.0):
        super(MeasurementNet, self).__init__()

        self.measurement_features = nn.Sequential(
            FcBlock(1, 512, dropout_rate=dropout_rate),
            FcBlock(512, 512, dropout_rate=dropout_rate)
        )
        

    def forward(self, measurements):
        x = self.measurement_features(measurements)
        return x
