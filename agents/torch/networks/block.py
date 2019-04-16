import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, batch_norm, dropout_rate=0.0):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=0)
        if batch_norm is True:
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=1e-3)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn'):
            x = self.bn(x)
        x = self.dropout(x)
        return self.activation(x)
    
    
class FcBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(FcBlock, self).__init__()
        
        self.lin = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.lin(x)
        x = self.dropout(x)
        return self.activation(x)