import torch
import torch.nn as nn
import torch.nn.functional as F

class SSA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SSA, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        batch_size, channels, height, width = inputs.size()
        q = k = v = self.conv(inputs.permute(0, 3, 1, 2))
        Qshape = q.size()
        Kshape = k.size()
        Vshape = v.size()
        a = Qshape[2] * Qshape[3]
        q = q.view(batch_size, Qshape[1], a)
        k = k.view(batch_size, Kshape[1], a).permute(0, 2, 1)
        qk = torch.matmul(q, k)
        qk = F.softmax(qk, dim=-1)
        v = v.view(batch_size, Vshape[1], a)
        qkv = torch.matmul(qk, v)
        qkv = qkv.view(batch_size, Vshape[1], Vshape[2], Vshape[3])
        qkv = qkv.permute(0, 1, 2, 3)
        qkv = self.conv(qkv)
        qkv = qkv.permute(0,3,2,1)
        return qkv

# Example usage:
inputs = torch.randn(1, 14, 14, 128)  # Assuming batch size of 1
ssa = SSA(in_channels=128, out_channels=128)
print("Input Shape", inputs.shape)
output = ssa(inputs)
print("Output shape:", output.shape)
