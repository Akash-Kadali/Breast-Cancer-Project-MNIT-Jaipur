import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias=False):
        super(Conv2dSamePadding, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, kernel_size=3, dilation=1, stride=1, bias=False):
        super(DepthwiseConv2d, self).__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class MSConv(nn.Module):
    def __init__(self, in_channels, fltr, stride):
        super(MSConv, self).__init__()
        self.depthwise1 = DepthwiseConv2d(in_channels, 1, in_channels, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.depthwise2 = DepthwiseConv2d(in_channels, 1, in_channels, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.depthwise3 = DepthwiseConv2d(in_channels, 1, in_channels, kernel_size=3, stride=stride)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.conv = Conv2dSamePadding(in_channels * 3, fltr, kernel_size=1)

    def forward(self, input):
        x = self.depthwise1(input)
        x = self.bn1(x)
        y = self.depthwise2(x)
        y = self.bn2(y)
        z = self.depthwise3(y)
        z = self.bn3(z)
        z = torch.cat([x, y, z], dim=1)
        z = self.conv(z)
        z = F.gelu(z)  # Replaced with GELU activation
        return z

class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = Conv2dSamePadding(in_channels=3, out_channels=32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = Conv2dSamePadding(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))  # Replaced with GELU activation
        x = F.gelu(self.bn2(self.conv2(x)))  # Replaced with GELU activation
        return x

class LKA(nn.Module):
    def __init__(self, in_channels, fltr):
        super(LKA, self).__init__()
        self.depthwise1 = DepthwiseConv2d(in_channels, 1, in_channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.depthwise2 = DepthwiseConv2d(in_channels, 1, in_channels, kernel_size=3, stride=1, dilation=3)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv = Conv2dSamePadding(in_channels, fltr, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = F.gelu(self.bn1(self.depthwise1(x)))  # Replaced with GELU activation
        x2 = F.gelu(self.bn2(self.depthwise2(x1)))  # Replaced with GELU activation
        x3 = F.gelu(self.conv(x2))  # Replaced with GELU activation
        out = x3 * x  # Use the input 'x' for the final multiplication
        return out

class GAB(nn.Module):
    def __init__(self, in_channels):
        super(GAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = Conv2dSamePadding(in_channels, in_channels, kernel_size=1)
        self.conv2 = Conv2dSamePadding(in_channels, in_channels, kernel_size=1)

    def forward(self, inputs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = F.gelu(x)  # Replaced with GELU activation
        x = self.conv2(x)
        x = torch.sigmoid(x)
        C_A = x * inputs
        x = torch.mean(C_A, dim=1, keepdim=True)
        x = torch.sigmoid(x)
        S_A = x * C_A
        return S_A

class DefConv(nn.Module):
    def __init__(self, in_channels, fltr, stride=1):
        super(DefConv, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = Conv2dSamePadding(in_channels, fltr, kernel_size=1)
        self.msconv = MSConv(fltr, fltr, stride)
        self.gab = GAB(fltr)
        self.conv2 = Conv2dSamePadding(fltr, fltr, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stride = stride
        self.conv_residual = Conv2dSamePadding(in_channels, fltr, kernel_size=1)

    def forward(self, input):
        x = self.bn1(input)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.msconv(x)
        x = self.gab(x)
        x = self.conv2(x)
        x = F.gelu(x)
        input = self.conv_residual(input)
        input = F.gelu(input)
        x = x + input
        return x

class TransEnc(nn.Module):
    def __init__(self, in_channels, fltr, nl, nh):
        super(TransEnc, self).__init__()
        self.nl = nl
        self.norm1 = nn.LayerNorm(fltr)
        self.norm2 = nn.LayerNorm(fltr)
        self.conv = Conv2dSamePadding(fltr, fltr, kernel_size=1)
        self.lka_attn = LKA(fltr, fltr)

    def forward(self, input):
        x = input
        for _ in range(self.nl):
            residual = x  # Store the input to add it later
            x = x.permute(0, 2, 3, 1)  # Change shape to (B, H, W, C) for LayerNorm
            x = self.norm1(x)
            x = x.permute(0, 3, 1, 2)  # Change back to (B, C, H, W) for Conv2d
            x = self.lka_attn(x)
            x = x + residual  # Add the residual connection
            y = x.permute(0, 2, 3, 1)  # Change shape to (B, H, W, C) for LayerNorm
            y = self.norm2(y)
            y = y.permute(0, 3, 1, 2)  # Change back to (B, C, H, W) for Conv2d
            y = self.conv(y)
            y = F.gelu(y)  # Replaced with GELU activation
            x = x + y  # Add the output of the convolution
        return x

class LKA_Attn(nn.Module):
    def __init__(self, in_channels, fltr, nh):
        super(LKA_Attn, self).__init__()
        self.nh = nh
        self.convs1 = nn.ModuleList([
            Conv2dSamePadding(in_channels, fltr, kernel_size=1) for _ in range(nh)
        ])
        self.lka_layers = nn.ModuleList([LKA(fltr, fltr) for _ in range(nh)])
        self.convs2 = nn.ModuleList([
            Conv2dSamePadding(fltr, fltr, kernel_size=1) for _ in range(nh)
        ])

    def forward(self, input):
        attn = []
        for i in range(self.nh):
            x = self.convs1[i](input)
            x = self.lka_layers[i](x)
            x = self.convs2[i](x)
            attn.append(x)

        mh_lka_attn = attn[0]
        for i in range(1, len(attn)):
            mh_lka_attn = mh_lka_attn + attn[i]
        return mh_lka_attn

class DefVitBlock(nn.Module):
    def __init__(self, in_channels, fltr, stride, nl, nh=1):
        super(DefVitBlock, self).__init__()
        self.defconv = DefConv(in_channels, fltr, stride)
        self.trans_enc = TransEnc(in_channels, fltr, nl, nh)

    def forward(self, input):
        x = self.defconv(input)
        x = self.trans_enc(x)
        return x

class DefVit(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(DefVit, self).__init__()
        self.stem = Stem()
        self.stage1_block1 = DefVitBlock(32, 32, stride=1, nl=2, nh=3)
        self.stage1_block2 = DefVitBlock(32, 32, stride=1, nl=2, nh=3)
        self.stage2_block1 = DefVitBlock(32, 64, stride=1, nl=2, nh=3)
        self.stage2_block2 = DefVitBlock(64, 64, stride=1, nl=2, nh=3)
        self.stage3_block1 = DefVitBlock(64, 128, stride=1, nl=4, nh=3)
        self.stage3_block2 = DefVitBlock(128, 128, stride=1, nl=4, nh=3)
        self.stage3_block3 = DefVitBlock(128, 128, stride=1, nl=4, nh=3)
        self.stage3_block4 = DefVitBlock(128, 128, stride=1, nl=4, nh=3)
        self.stage3_block5 = DefVitBlock(128, 128, stride=1, nl=4, nh=3)
        self.stage4_block1 = DefVitBlock(128, 128, stride=1, nl=2, nh=3)
        self.stage4_block2 = DefVitBlock(128, 128, stride=1, nl=2, nh=3)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.stage3_block3(x)
        x = self.stage3_block4(x)
        x = self.stage3_block5(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.stage4_block1(x)
        x = self.stage4_block2(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Example usage:
input_tensor = torch.randn(1, 3, 224, 224)
num_classes = 8
model = DefVit(input_shape=(3, 224, 224), num_classes=num_classes)
output = model(input_tensor)
print(output.shape)
