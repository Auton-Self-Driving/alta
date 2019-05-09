import torch
import torch.nn as nn

from .block import ConvBlock, FcBlock
from .abstract_cnn import CNN


class DrivingLargeCNN(CNN):
    """
    1175712 + 4457472 = 5633184 parameters
    Input: 
        - image: tensor of size (batch_size, 3, 88, 200)
    Output:
        - image_features: tensor of size (batch_size, 512)
    """
    def __init__(self, batch_norm, dropout_rate=0.0):
        super(CNN, self).__init__()
        
        # Convolutional core
        # in_channels, out_channels, kernel_size, stride
        self.core = nn.Sequential(
            ConvBlock(3, 32, 5, 2, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(32, 32, 3, 1, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(32, 64, 3, 2, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(64, 64, 3, 1, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(64, 128, 3, 2, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(128, 128, 3, 1, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(128, 256, 3, 1, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(256, 256, 3, 1, batch_norm, dropout_rate=dropout_rate)
        )

        # Fully connected layers 
        self.fc = nn.Sequential(
            FcBlock(8192, 512),
            FcBlock(512, 512)
        )
        
    def forward(self, image):
        x = self.core(image)
        # Reshape to Tensorflow format
        # For compatibility with weights pretrained by imitation learning
        #x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc(x.view(x.size(0), -1))
        return x
    

class DrivingSmallCNN(CNN):
    """
    187424 + 754688 = 942112 parameters
    Input: 
        - image: tensor of size (batch_size, 3, 88, 200)
    Output:
        - image_features: tensor of size (batch_size, 512)
    """
    def __init__(self, batch_norm, dropout_rate=0.0):
        super(CNN, self).__init__()
        
        # Convolutional core
        # in_channels, out_channels, kernel_size, stride
        self.core = nn.Sequential(
            ConvBlock(3, 32, 8, 4, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(32, 64, 4, 2, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(64, 64, 3, 1, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(64, 64, 3, 1, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(64, 64, 3, 1, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(64, 64, 3, 1, batch_norm, dropout_rate=dropout_rate)
        )
       
        # Fully connected layers 
        self.fc = nn.Sequential(
            FcBlock(64 * 15, 512),
            FcBlock(512, 512)
        )
        
    def forward(self, image):
        x = self.core(image)
        x = self.fc(x.view(x.size(0), -1))
        return x


class DrivingSmallestCNN(CNN):
    """
    # 187424 + 754688 = 942112 parameters
    Input: 
        - image: tensor of size (batch_size, 3, 88, 200)
    Output:
        - image_features: tensor of size (batch_size, 512)
    """

    def __init__(self, batch_norm, dropout_rate=0.0):
        super(CNN, self).__init__()

        # Convolutional core
        # in_channels, out_channels, kernel_size, stride
        self.core = nn.Sequential(
            ConvBlock(3, 16, 5, 3, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(16, 32, 5, 3, batch_norm, dropout_rate=dropout_rate),
            ConvBlock(32, 64, 5, 3, batch_norm, dropout_rate=dropout_rate)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            FcBlock(768, 512),
            FcBlock(512, 512)
        )

    def forward(self, image):
        x = self.core(image)
        x = self.fc(x.view(x.size(0), -1))
        return x
