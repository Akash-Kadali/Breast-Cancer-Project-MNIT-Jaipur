import torch
import torch.nn as nn
import torch.nn.functional as F

class SSA(nn.Module):
    def __init__(self, fltr):
        super(SSA, self).__init__()
        self.conv = nn.Conv2d(in_channels=fltr, out_channels=fltr, kernel_size=1, stride=1, padding=0) 
        
    def forward(self, inputs):
        shape = inputs.shape
        q = k = v = self.conv(inputs)
        
        Qshape = q.shape
        Kshape = k.shape
        Vshape = v.shape
        print("Input Q, K, V shape:", Qshape, Kshape, Vshape)
        
        a = Qshape[2] * Qshape[3]  # height * width
        q = q.reshape(q.shape[0], q.shape[1], a)
        print("After reshaping Query:", q.shape)
        
        k = k.reshape(k.shape[0], k.shape[1], a)
        k = k.permute(0, 2, 1)
        print("After reshaping Key:", k.shape)
        
        qk = torch.matmul(q, k)
        qk = F.softmax(qk, dim=-1)
        print("Dot product of Q and K:", qk.shape)
        
        v = v.reshape(v.shape[0], v.shape[1], a)
        print("After reshaping Value:", v.shape)
        
        qkv = torch.matmul(qk, v)
        print(qkv.shape)
        
        qkv = qkv.reshape(shape[0], shape[1], shape[2], shape[3])
        qkv = self.conv(qkv)
        return qkv

# Example usage
model = SSA(fltr=256)  # corrected to remove 'in_channels' and specify filter size
inputs = torch.randn(1, 256, 14, 14)  # corrected input shape to (batch, channels, height, width)
output = model(inputs)
print("Output Shape",output.shape)