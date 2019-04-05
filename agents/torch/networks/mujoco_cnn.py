import torch
import torch.nn as nn

from .block import ConvBlock, FcBlock
from .abstract_cnn import CNN


class MujocoLargeCNN(CNN):
    """
    898480 parameters
    Input: 
        - image: tensor of size (batch_size, 3, 84, 84)
    Output:
        - image_features: tensors of size (batch_size, 400)
    """
    def __init__(self, batch_norm):
        super(CNN, self).__init__()
        
        # Convolutional core
        # in_channels, out_channels, kernel_size, stride
        self.core = nn.Sequential(
            ConvBlock(3, 32, 5, 2, batch_norm),
            ConvBlock(32, 32, 3, 1, batch_norm),
            ConvBlock(32, 64, 3, 2, batch_norm),
            ConvBlock(64, 64, 3, 1, batch_norm),
            ConvBlock(64, 128, 3, 2, batch_norm),
            ConvBlock(128, 128, 3, 1, batch_norm),
            ConvBlock(128, 128, 3, 1, batch_norm)
        )

        # Fully connected layer
        self.fc = FcBlock(128 * 3 * 3, 400)
        
    def forward(self, image):
        x = self.core(image)
        x = self.fc(x.view(x.size(0), -1))
        return x

    
class MujocoSmallCNN(CNN):
    """
    286480 parameters
    Input: 
        - image: tensor of size (batch_size, 3, 84, 84)
    Output:
        - image_features: tensors of size (batch_size, 400)
    """
    def __init__(self, batch_norm):
        super(CNN, self).__init__()
        
        # Convolutional core
        # in_channels, out_channels, kernel_size, stride
        self.core = nn.Sequential(
            ConvBlock(3, 32, 8, 4, batch_norm),
            ConvBlock(32, 32, 4, 2, batch_norm),
            ConvBlock(32, 64, 4, 2, batch_norm),
        )

        # Fully connected layer
        self.fc = FcBlock(64 * 3 * 3, 400)
        
    def forward(self, image):
        x = self.core(image)
        x = self.fc(x.view(x.size(0), -1))
        return x