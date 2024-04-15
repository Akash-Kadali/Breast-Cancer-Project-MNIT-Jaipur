class CDSA(nn.Module):
    def __init__(self, fltr, nh):
        super(CDSA, self).__init__()
        self.fltr = fltr
        self.nh = nh
        self.attn = nn.ModuleList([SSA(fltr // nh, fltr // nh) for _ in range(nh)])  # Self-Attention modules
        self.conv = nn.Conv2d(fltr, fltr, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        attn = []
        feature_split = torch.chunk(input_tensor, self.nh, dim=3)  # Split input tensor along channel dimension
        shape = feature_split[0].shape
        
        # Apply self-attention to each split of the input tensor
        x = self.attn[0](feature_split[0])
        attn.append(x)
        
        for i in range(1, self.nh):
            x = feature_split[i] + x  # Residual connection
            x = self.attn[i](x)  # Apply self-attention
            attn.append(x)
        
        # Concatenate the outputs of self-attention along the channel dimension
        mh_lka_attn = torch.cat(attn, dim=3)
        
        # Apply 1x1 convolution followed by ReLU activation
        mh_lka_attn = mh_lka_attn.permute(0,3,2,1)
        mh_lka_attn = self.conv(mh_lka_attn)
        mh_lka_attn = self.relu(mh_lka_attn)
        mh_lka_attn = mh_lka_attn.permute(0,3,2,1)
        return mh_lka_attn