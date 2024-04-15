def CAL(input, fltr, nh):
    print("Shape of CAL Input", input.shape)
    x = input
    rs1 = x = torch.add(x, input)
    x = nn.LayerNorm(x.size()[1:], eps=1e-6)(x)
    cdsa = CDSA(fltr, nh)
    cdsa_output = cdsa(x)
    x = cdsa_output 
    rs2 = x = torch.add(rs1, x)
    x = nn.LayerNorm(x.size()[1:], eps=1e-6)(x)
    x = x.permute(0, 3, 2, 1)
    x = nn.Conv2d(fltr, fltr, kernel_size=1, padding='same')(x)
    x = x.permute(0, 3, 2, 1)
    x = torch.add(rs2, x)
    print("Shape of CAL Output", x.shape)
    return x