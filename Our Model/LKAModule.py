import torch
import torch.nn as nn

class LKAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LKAModule, self).__init__()
        
        # 2D Convolutional Layer
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # GELU activation function
        self.gelu = nn.GELU()
        
        # Depthwise Separable Convolution 3x3
        self.dwconv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        
        # Depthwise Separable Convolution 3x3, stride=2
        self.dwconv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        
        # 1x1 Convolutional Layer
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        # Additional 2D Convolutional Layer
        self.conv2d_additional = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 2D Convolution
        a = self.conv2d(x)
        
        # GELU activation
        b = self.gelu(a)
        
        # Depthwise Separable Convolution 3x3
        c = self.dwconv3x3(b)
        
        # Depthwise Separable Convolution 3x3, stride=2
        d = self.dwconv3x3_2(c)
        
        # 1x1 Convolution
        e = self.conv1x1(d)
        
        # Additional 2D Convolution
        f = self.conv2d_additional(e^x)
        
        g = f^b
        
        return g
